import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from experiment_reproducibility import (
    find_latest_run,
    installed_versions,
    sha256_file,
    sha256_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_ROOT
    / "Scenario B-ARFF"
    / "TimeBasedFeatures-Dataset-15s.csv"
)
RUNS_ROOT = PROJECT_ROOT / "Scenario B-ARFF" / "reproducible_runs"

QOS_POLICY = {
    "VOIP": ("HIGH", 1, "Telephony", "EF", 46),
    "VPN-VOIP": ("HIGH", 1, "Telephony", "EF", 46),
    "STREAMING": ("HIGH", 2, "Multimedia Streaming", "AF31", 26),
    "VPN-STREAMING": ("HIGH", 2, "Multimedia Streaming", "AF31", 26),
    "BROWSING": ("MEDIUM", 3, "Low-Latency Data", "AF21", 18),
    "VPN-BROWSING": ("MEDIUM", 3, "Low-Latency Data", "AF21", 18),
    "CHAT": ("MEDIUM", 3, "Low-Latency Data", "AF21", 18),
    "VPN-CHAT": ("MEDIUM", 3, "Low-Latency Data", "AF21", 18),
    "MAIL": ("LOW", 4, "High-Throughput Data", "AF11", 10),
    "VPN-MAIL": ("LOW", 4, "High-Throughput Data", "AF11", 10),
    "FT": ("LOW", 4, "Low-Priority Data", "CS1", 8),
    "VPN-FT": ("LOW", 4, "Low-Priority Data", "CS1", 8),
    "P2P": ("LOW", 4, "Low-Priority Data", "CS1", 8),
    "VPN-P2P": ("LOW", 4, "Low-Priority Data", "CS1", 8),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate QoS assignments from a saved clean-XGBoost run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Experiment run directory; defaults to the latest completed run.",
    )
    return parser.parse_args()


def load_and_validate_run(run_directory):
    run_directory = Path(run_directory).resolve()
    manifest_path = run_directory / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    actual_dataset_hash = sha256_file(DATASET_PATH)
    expected_dataset_hash = manifest["dataset"]["sha256"]
    if actual_dataset_hash != expected_dataset_hash:
        raise ValueError(
            "Dataset SHA-256 does not match the saved experiment. "
            f"Expected {expected_dataset_hash}, got {actual_dataset_hash}."
        )

    environment = manifest["environment"]
    current_versions = installed_versions()
    mismatches = {
        package: (expected, current_versions.get(package))
        for package, expected in environment.items()
        if current_versions.get(package) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{package}: expected {expected}, found {actual}"
            for package, (expected, actual) in mismatches.items()
        )
        raise RuntimeError(
            "Runtime versions differ from the saved experiment: "
            f"{details}. Install the pinned packages from ml-requirements.txt."
        )

    df = pd.read_csv(DATASET_PATH)
    feature_columns = df.columns[:-1].tolist()
    target_column = df.columns[-1]
    expected_features = manifest["preprocessing"]["feature_columns"]
    expected_target = manifest["preprocessing"]["target_column"]
    if feature_columns != expected_features:
        raise ValueError(
            "Dataset feature columns or order do not match the saved model."
        )
    if target_column != expected_target:
        raise ValueError(
            f"Dataset target column changed: expected {expected_target!r}, "
            f"got {target_column!r}."
        )

    class_mapping_path = run_directory / manifest["label_mapping_file"]
    if sha256_file(class_mapping_path) != manifest["label_mapping_sha256"]:
        raise ValueError("Saved label mapping hash does not match the manifest.")
    mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    classes = mapping["classes"]
    class_to_index = mapping["class_to_index"]
    expected_mapping = {
        label: index for index, label in enumerate(classes)
    }
    if class_to_index != expected_mapping:
        raise ValueError("Saved LabelEncoder class mapping is invalid.")

    split_path = run_directory / manifest["split"]["index_file"]
    with np.load(split_path, allow_pickle=False) as split:
        clean_source_indices = split["clean_source_indices"]
        train_source_indices = split["train_source_indices"]
        test_source_indices = split["test_source_indices"]

    for name, indices in (
        ("clean", clean_source_indices),
        ("train", train_source_indices),
        ("test", test_source_indices),
    ):
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Saved {name} row indices are invalid.")
        if np.any(indices < 0) or np.any(indices >= len(df)):
            raise ValueError(f"Saved {name} row indices exceed the dataset.")

    clean_set = set(clean_source_indices.tolist())
    train_set = set(train_source_indices.tolist())
    test_set = set(test_source_indices.tolist())
    if not train_set.issubset(clean_set) or not test_set.issubset(clean_set):
        raise ValueError("Saved split contains rows excluded from clean data.")
    if train_set.intersection(test_set):
        raise ValueError("Saved train and test indices overlap.")
    if len(train_source_indices) != manifest["split"]["train_row_count"]:
        raise ValueError("Saved train index count does not match manifest.")
    if len(test_source_indices) != manifest["split"]["test_row_count"]:
        raise ValueError("Saved test index count does not match manifest.")
    if sha256_indices(train_source_indices) != (
        manifest["split"]["train_source_indices_sha256"]
    ):
        raise ValueError("Saved train indices do not match the manifest.")
    if sha256_indices(test_source_indices) != (
        manifest["split"]["test_source_indices_sha256"]
    ):
        raise ValueError("Saved test indices do not match the manifest.")

    model_path = run_directory / manifest["model"]["artifact_file"]
    if sha256_file(model_path) != manifest["model"]["sha256"]:
        raise ValueError("Saved model hash does not match the experiment manifest.")
    model = joblib.load(model_path)
    imputer = model.named_steps.get("imputer")
    classifier = model.named_steps.get("classifier")
    if imputer is None or classifier is None:
        raise ValueError("Saved model does not have the expected pipeline steps.")
    model_features = getattr(imputer, "feature_names_in_", None)
    if model_features is None or list(model_features) != expected_features:
        raise ValueError("Saved model feature columns do not match the dataset.")
    actual_parameters = classifier.get_params()
    parameter_mismatches = {
        name: (expected, actual_parameters.get(name))
        for name, expected in manifest["model"]["xgboost_parameters"].items()
        if actual_parameters.get(name) != expected
    }
    if parameter_mismatches:
        raise ValueError("Saved model parameters do not match the manifest.")
    if classifier.n_features_in_ != len(expected_features):
        raise ValueError("Saved model feature count does not match the dataset.")
    if not np.array_equal(
        classifier.classes_,
        np.arange(len(classes)),
    ):
        raise ValueError("Saved model class IDs do not match the label mapping.")

    return (
        manifest,
        mapping,
        model,
        df,
        train_source_indices,
        test_source_indices,
    )


def unique_policy_output_path(run_directory):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = run_directory / f"qos_policy_output_{timestamp}.csv"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = run_directory / f"qos_policy_output_{timestamp}_{suffix}.csv"
        suffix += 1
    return candidate


def main():
    args = parse_args()
    run_directory = (
        args.run_dir.resolve()
        if args.run_dir
        else find_latest_run(RUNS_ROOT)
    )
    (
        manifest,
        mapping,
        model,
        df,
        _train_source_indices,
        test_source_indices,
    ) = load_and_validate_run(run_directory)

    feature_columns = manifest["preprocessing"]["feature_columns"]
    X_test = (
        df.iloc[test_source_indices]
        .loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    y_test = df.iloc[test_source_indices, -1].astype(str)
    class_to_index = mapping["class_to_index"]
    unknown_labels = sorted(set(y_test) - set(class_to_index))
    if unknown_labels:
        raise ValueError(f"Dataset contains unmapped labels: {unknown_labels}")

    predicted_ids = model.predict(X_test).astype(int)
    index_to_class = {
        index: label for label, index in class_to_index.items()
    }
    unknown_predictions = sorted(set(predicted_ids) - set(index_to_class))
    if unknown_predictions:
        raise ValueError(
            f"Model returned unmapped class IDs: {unknown_predictions}"
        )
    predicted_classes = [index_to_class[index] for index in predicted_ids]
    missing_policy_classes = sorted(set(predicted_classes) - set(QOS_POLICY))
    if missing_policy_classes:
        raise ValueError(
            f"QoS policy has no mapping for: {missing_policy_classes}"
        )

    results = pd.DataFrame({
        "Actual_Traffic": y_test.to_numpy(),
        "Predicted_Traffic": predicted_classes,
    })
    results["Prediction_Correct"] = (
        results["Actual_Traffic"] == results["Predicted_Traffic"]
    )
    results["Priority"] = results["Predicted_Traffic"].map(
        lambda label: QOS_POLICY[label][0]
    )
    results["Queue_Number"] = results["Predicted_Traffic"].map(
        lambda label: QOS_POLICY[label][1]
    )
    results["Service_Class"] = results["Predicted_Traffic"].map(
        lambda label: QOS_POLICY[label][2]
    )
    results["DSCP_Name"] = results["Predicted_Traffic"].map(
        lambda label: QOS_POLICY[label][3]
    )
    results["DSCP_Value"] = results["Predicted_Traffic"].map(
        lambda label: QOS_POLICY[label][4]
    )

    output_file = unique_policy_output_path(run_directory)
    results.to_csv(output_file, index=False)

    print("\n========== QoS RESULTS ==========")
    print("Experiment run:", manifest["run_id"])
    print("Total test flows:", len(results))
    print(
        "Classification accuracy:",
        round(results["Prediction_Correct"].mean() * 100, 2),
        "%",
    )
    print("\nPriority distribution:")
    print(results["Priority"].value_counts())
    print("\nQoS output saved:")
    print(output_file)


if __name__ == "__main__":
    main()
