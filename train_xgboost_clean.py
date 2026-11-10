import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from experiment_reproducibility import (
    create_run_directory,
    installed_versions,
    load_clean_dataset,
    sha256_file,
    sha256_indices,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_ROOT
    / "Scenario B-ARFF"
    / "TimeBasedFeatures-Dataset-15s.csv"
)
RUNS_ROOT = PROJECT_ROOT / "Scenario B-ARFF" / "reproducible_runs"

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset_hash = sha256_file(DATASET_PATH)
    experiment = load_clean_dataset(DATASET_PATH)
    X = experiment["X"]
    y_encoded = experiment["y_encoded"]
    train_indices = experiment["train_indices"]
    test_indices = experiment["test_indices"]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y_encoded[train_indices]
    y_test = y_encoded[test_indices]

    classifier_params = {
        **XGBOOST_PARAMS,
        "num_class": len(experiment["encoder"].classes_),
    }
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(**classifier_params)),
    ])

    run_directory = create_run_directory(RUNS_ROOT)
    print("Original records:", len(experiment["raw_df"]))
    print("Removed ambiguous records:", experiment["removed_row_count"])
    print("Clean records:", len(X))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("\nTraining XGBoost...")

    start = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = time.time() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(experiment["encoder"].classes_)),
        target_names=experiment["encoder"].classes_,
        zero_division=0,
    )

    model_path = run_directory / "xgboost_clean.joblib"
    joblib.dump(model, model_path)
    report_path = run_directory / "clean_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    split_path = run_directory / "split_indices.npz"
    np.savez_compressed(
        split_path,
        clean_source_indices=experiment["clean_source_indices"],
        train_source_indices=experiment["train_source_indices"],
        test_source_indices=experiment["test_source_indices"],
    )

    classes = experiment["encoder"].classes_.tolist()
    class_mapping = {
        "classes": classes,
        "class_to_index": {
            label: index for index, label in enumerate(classes)
        },
    }
    mapping_path = run_directory / "label_mapping.json"
    write_json(mapping_path, class_mapping)

    manifest = {
        "schema_version": 1,
        "run_id": run_directory.name,
        "dataset": {
            "filename": DATASET_PATH.name,
            "sha256": dataset_hash,
            "raw_row_count": int(len(experiment["raw_df"])),
            "clean_row_count": int(len(X)),
            "conflicting_feature_group_count": (
                experiment["conflicting_group_count"]
            ),
            "removed_conflicting_row_count": (
                experiment["removed_row_count"]
            ),
        },
        "preprocessing": {
            "target_column": experiment["target_column"],
            "feature_columns": experiment["feature_columns"],
            "feature_conversion": "pandas.to_numeric(errors='coerce')",
            "infinite_value_handling": "replace with NaN",
            "conflicting_label_handling": (
                "Remove all rows with identical feature hashes "
                "having more than one distinct label"
            ),
            "imputation": "SimpleImputer(strategy='median') fitted on train",
        },
        "split": {
            "class": "StratifiedGroupKFold",
            "n_splits": 5,
            "fold_index": 0,
            "shuffle": True,
            "random_state": 42,
            "index_file": split_path.name,
            "index_space": (
                "zero-based CSV data-row positions; header excluded"
            ),
            "train_row_count": int(len(train_indices)),
            "test_row_count": int(len(test_indices)),
            "train_source_indices_sha256": sha256_indices(
                experiment["train_source_indices"]
            ),
            "test_source_indices_sha256": sha256_indices(
                experiment["test_source_indices"]
            ),
        },
        "label_mapping_file": mapping_path.name,
        "label_mapping_sha256": sha256_file(mapping_path),
        "model": {
            "artifact_file": model_path.name,
            "sha256": sha256_file(model_path),
            "pipeline": [
                "SimpleImputer(strategy='median')",
                "XGBClassifier",
            ],
            "xgboost_parameters": classifier_params,
        },
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "training_and_prediction_seconds": round(elapsed, 6),
        },
        "environment": installed_versions(),
    }
    manifest_path = run_directory / "experiment_manifest.json"
    write_json(manifest_path, manifest)

    print("\nACCURACY:", round(accuracy * 100, 2), "%")
    print("MACRO F1:", round(macro_f1, 4))
    print("WEIGHTED F1:", round(weighted_f1, 4))
    print("\nCLASSIFICATION REPORT:\n")
    print(report)
    print("Run directory:", run_directory)
    print("Manifest:", manifest_path)
    print("Model:", model_path)
    print("Elapsed seconds:", round(elapsed, 2))
    print("\nRun generate_qos_policy.py to use this saved experiment.")


if __name__ == "__main__":
    main()
