import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from calibration_models import TemperatureScaledModel
from experiment_reproducibility import sha256_file, sha256_indices, write_json
from generate_qos_policy import QOS_POLICY


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "Scenario B-ARFF"
    / "reproducible_runs"
    / "clean_xgboost_20261101T173109339548Z"
)
QOS_COMPARE_PATH = PROJECT_ROOT / "Scenario B-ARFF" / "qos_compare.py"
SPLIT_SEED = 2028
BOOTSTRAP_SEED = 2029
BOOTSTRAP_REPLICATES = 500
SPLIT_COUNT = 5
MODEL_FOLDS = (0, 1, 2)
CALIBRATION_FOLD = 3
POLICY_FOLD = 4
TARGET_COVERAGES = (0.50, 0.70, 0.90)
MIN_CLASS_POLICY_SUPPORT = 30
CALIBRATION_BINS = 15
ECE_BIN_EDGES = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
PRIORITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare leakage-free confidence-aware QoS decisions "
            "using the saved clean-XGBoost experiment."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Immutable source experiment run directory.",
    )
    return parser.parse_args()


def create_experiment_directory(run_directory):
    root = Path(run_directory) / "qos_decision_experiments"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = f"phase4_{timestamp}"
    candidate = root / base
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = root / f"{base}_{suffix}"
            suffix += 1


def installed_versions():
    distributions = (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
        "scipy",
        "threadpoolctl",
    )
    result = {"python": platform.python_version()}
    for package in distributions:
        try:
            result[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Required package is not installed: {package}"
            ) from error
    return result


def multiclass_ece(y_true, probabilities):
    ece_values = []
    for class_index in range(probabilities.shape[1]):
        class_probabilities = probabilities[:, class_index]
        class_targets = (y_true == class_index).astype(float)
        bin_ids = np.digitize(
            class_probabilities,
            ECE_BIN_EDGES[1:-1],
            right=False,
        )
        class_ece = 0.0
        for bin_index in range(CALIBRATION_BINS):
            mask = bin_ids == bin_index
            if mask.any():
                class_ece += float(mask.mean()) * abs(
                    float(class_probabilities[mask].mean())
                    - float(class_targets[mask].mean())
                )
        ece_values.append(class_ece)
    return float(np.mean(ece_values))


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Classifier returned invalid probability values.")
    if np.any(probabilities < 0):
        raise ValueError("Classifier returned negative class probabilities.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Classifier returned a probability row with zero mass.")
    return probabilities / row_sums


def top_label_ece(y_true, predicted, confidence):
    bin_ids = np.digitize(
        confidence,
        ECE_BIN_EDGES[1:-1],
        right=False,
    )
    result = 0.0
    for bin_index in range(CALIBRATION_BINS):
        mask = bin_ids == bin_index
        if mask.any():
            result += float(mask.mean()) * abs(
                float(confidence[mask].mean())
                - float((predicted[mask] == y_true[mask]).mean())
            )
    return float(result)


def probability_metrics(y_true, probabilities):
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(
            y_true,
            predicted,
            labels=np.arange(probabilities.shape[1]),
            average="macro",
            zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            y_true,
            predicted,
            labels=np.arange(probabilities.shape[1]),
            average="weighted",
            zero_division=0,
        )),
        "multiclass_brier": brier,
        "normalized_brier": brier / 2.0,
        "multiclass_log_loss": float(log_loss(
            y_true,
            probabilities,
            labels=np.arange(probabilities.shape[1]),
        )),
        "top_label_ece": top_label_ece(y_true, predicted, confidence),
        "classwise_ece": multiclass_ece(y_true, probabilities),
    }


def find_threshold(values, target_coverage):
    if len(values) == 0:
        raise ValueError("Cannot select a threshold from an empty partition.")
    return float(np.quantile(
        np.asarray(values, dtype=float),
        1.0 - target_coverage,
        method="lower",
    ))


def build_thresholds(
    method,
    policy_probabilities,
    policy_labels,
    class_names,
):
    predicted = policy_probabilities.argmax(axis=1)
    confidence = policy_probabilities.max(axis=1)
    result = {}
    for target_coverage in TARGET_COVERAGES:
        global_threshold = find_threshold(confidence, target_coverage)
        global_acceptance = confidence >= global_threshold
        class_thresholds = {}
        class_support = {}
        fallback_classes = []
        for class_index, class_name in enumerate(class_names):
            mask = predicted == class_index
            support = int(mask.sum())
            class_support[class_name] = support
            if support >= MIN_CLASS_POLICY_SUPPORT:
                class_thresholds[class_name] = find_threshold(
                    confidence[mask],
                    target_coverage,
                )
            else:
                class_thresholds[class_name] = global_threshold
                fallback_classes.append(class_name)
        per_class_thresholds = np.asarray([
            class_thresholds[class_names[index]]
            for index in predicted
        ])
        class_acceptance = confidence >= per_class_thresholds
        result[f"{target_coverage:.2f}"] = {
            "target_coverage": target_coverage,
            "global_threshold": global_threshold,
            "global_policy_accepted_rows": int(global_acceptance.sum()),
            "global_policy_rows": int(len(policy_labels)),
            "global_policy_achieved_coverage": float(
                global_acceptance.mean()
            ),
            "class_thresholds": class_thresholds,
            "class_support": class_support,
            "class_threshold_fallback_classes": fallback_classes,
            "class_policy_accepted_rows": int(class_acceptance.sum()),
            "class_policy_rows": int(len(policy_labels)),
            "class_policy_achieved_coverage": float(class_acceptance.mean()),
        }
    return result


def policy_metrics(
    y_true,
    predicted,
    confidence,
    accepted,
    assigned_class_names,
    class_names,
    probabilities,
):
    correct = predicted == y_true
    actual_priority = np.asarray([
        PRIORITY_RANK[QOS_POLICY[class_names[index]][0]]
        for index in y_true
    ])
    assigned_priority = np.asarray([
        PRIORITY_RANK[QOS_POLICY[label][0]]
        for label in assigned_class_names
    ])
    error_mask = ~correct
    accepted_count = int(accepted.sum())
    error_count = int(error_mask.sum())
    accepted_error_count = int((accepted & error_mask).sum())
    upward = assigned_priority > actual_priority
    downward = assigned_priority < actual_priority
    unjustified_high = (assigned_priority == 3) & (actual_priority < 3)
    result = probability_metrics(y_true, probabilities)
    result.update({
        "accepted_rows": accepted_count,
        "accepted_coverage": float(accepted.mean()),
        "abstained_rows": int((~accepted).sum()),
        "selective_accuracy": (
            float(correct[accepted].mean()) if accepted_count else None
        ),
        "selective_macro_f1": (
            float(f1_score(
                y_true[accepted],
                predicted[accepted],
                labels=np.arange(len(class_names)),
                average="macro",
                zero_division=0,
            ))
            if accepted_count else None
        ),
        "selective_weighted_f1": (
            float(f1_score(
                y_true[accepted],
                predicted[accepted],
                labels=np.arange(len(class_names)),
                average="weighted",
                zero_division=0,
            ))
            if accepted_count else None
        ),
        "false_accept_rate_among_errors": (
            float(accepted_error_count / error_count)
            if error_count else None
        ),
        "accepted_error_rate": (
            float(accepted_error_count / accepted_count)
            if accepted_count else None
        ),
        "incorrect_high_priority_assignment_rate": float(
            unjustified_high.mean()
        ),
        "upward_priority_error_rate": float(upward.mean()),
        "downward_priority_error_rate": float(downward.mean()),
        "assigned_high_priority_rows": int((assigned_priority == 3).sum()),
        "unjustified_high_priority_rows": int(unjustified_high.sum()),
    })
    return result, {
        "correct": correct,
        "accepted": accepted,
        "errors": error_mask,
        "unjustified_high": unjustified_high,
        "upward": upward,
        "downward": downward,
    }


def class_reliability_rows(
    method,
    policy_name,
    y_true,
    predicted,
    confidence,
    accepted,
    probabilities,
    assigned_priority,
    class_names,
):
    rows = []
    for class_index, class_name in enumerate(class_names):
        actual = y_true == class_index
        predicted_as_class = predicted == class_index
        class_ece = 0.0
        class_probabilities = probabilities[:, class_index]
        targets = actual.astype(float)
        bin_ids = np.digitize(
            class_probabilities,
            ECE_BIN_EDGES[1:-1],
            right=False,
        )
        for bin_index in range(CALIBRATION_BINS):
            in_bin = bin_ids == bin_index
            if in_bin.any():
                class_ece += float(in_bin.mean()) * abs(
                    float(class_probabilities[in_bin].mean())
                    - float(targets[in_bin].mean())
                )
        actual_rank = PRIORITY_RANK[QOS_POLICY[class_name][0]]
        rows.append({
            "method": method,
            "policy": policy_name,
            "traffic_class": class_name,
            "actual_samples": int(actual.sum()),
            "prediction_recall": (
                float((predicted[actual] == class_index).mean())
                if actual.any() else None
            ),
            "predicted_class_precision": (
                float((predicted[predicted_as_class] == y_true[predicted_as_class]).mean())
                if predicted_as_class.any() else None
            ),
            "accepted_coverage": (
                float(accepted[actual].mean()) if actual.any() else None
            ),
            "accepted_accuracy": (
                float((predicted[actual & accepted] == y_true[actual & accepted]).mean())
                if (actual & accepted).any() else None
            ),
            "mean_probability_for_class": float(
                class_probabilities.mean()
            ),
            "one_vs_rest_brier": float(np.mean(
                (class_probabilities - targets) ** 2
            )),
            "classwise_ece": class_ece,
            "actual_priority_rank": actual_rank,
            "mean_assigned_priority_rank": (
                float(assigned_priority[actual].mean()) if actual.any() else None
            ),
            "actual_class_samples_predicted_as_class": int(
                predicted_as_class[actual].sum()
            ),
            "mean_confidence_when_predicted_as_class": (
                float(confidence[predicted_as_class].mean())
                if predicted_as_class.any() else None
            ),
        })
    return rows


def cluster_bootstrap_intervals(
    scenarios,
    group_codes,
    group_count,
    seed,
):
    rng = np.random.default_rng(seed)
    rows = []
    metric_names = (
        "accepted_coverage",
        "selective_accuracy",
        "false_accept_rate_among_errors",
        "incorrect_high_priority_assignment_rate",
        "upward_priority_error_rate",
    )
    for scenario in scenarios:
        masks = scenario["masks"]
        grouped = {
            "rows": np.bincount(
                group_codes,
                minlength=group_count,
            ).astype(float),
            "accepted": np.bincount(
                group_codes,
                weights=masks["accepted"].astype(float),
                minlength=group_count,
            ),
            "correct_accepted": np.bincount(
                group_codes,
                weights=(masks["accepted"] & masks["correct"]).astype(float),
                minlength=group_count,
            ),
            "errors": np.bincount(
                group_codes,
                weights=masks["errors"].astype(float),
                minlength=group_count,
            ),
            "accepted_errors": np.bincount(
                group_codes,
                weights=(masks["accepted"] & masks["errors"]).astype(float),
                minlength=group_count,
            ),
            "unjustified_high": np.bincount(
                group_codes,
                weights=masks["unjustified_high"].astype(float),
                minlength=group_count,
            ),
            "upward": np.bincount(
                group_codes,
                weights=masks["upward"].astype(float),
                minlength=group_count,
            ),
        }
        samples = {name: [] for name in metric_names}
        for _ in range(BOOTSTRAP_REPLICATES):
            selected = rng.integers(0, group_count, size=group_count)
            sampled_rows = float(grouped["rows"][selected].sum())
            accepted_count = float(grouped["accepted"][selected].sum())
            error_count = float(grouped["errors"][selected].sum())
            accepted_error_count = float(
                grouped["accepted_errors"][selected].sum()
            )
            samples["accepted_coverage"].append(
                accepted_count / sampled_rows if sampled_rows else np.nan
            )
            samples["selective_accuracy"].append(
                float(grouped["correct_accepted"][selected].sum())
                / accepted_count
                if accepted_count else np.nan
            )
            samples["false_accept_rate_among_errors"].append(
                accepted_error_count / error_count
                if error_count else np.nan
            )
            samples["incorrect_high_priority_assignment_rate"].append(
                float(grouped["unjustified_high"][selected].sum())
                / sampled_rows if sampled_rows else np.nan
            )
            samples["upward_priority_error_rate"].append(
                float(grouped["upward"][selected].sum())
                / sampled_rows if sampled_rows else np.nan
            )
        for metric in metric_names:
            valid = np.asarray(samples[metric], dtype=float)
            valid = valid[np.isfinite(valid)]
            point = scenario["metrics"].get(metric)
            rows.append({
                "method": scenario["method"],
                "policy": scenario["policy"],
                "metric": metric,
                "point_estimate": point,
                "ci_95_lower": (
                    float(np.quantile(valid, 0.025)) if len(valid) else None
                ),
                "ci_95_upper": (
                    float(np.quantile(valid, 0.975)) if len(valid) else None
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": seed,
                "resampling_unit": "exact-feature group",
            })
    return rows


def paired_cluster_bootstrap_deltas(
    scenarios,
    group_codes,
    group_count,
    seed,
):
    rng = np.random.default_rng(seed)
    metric_names = (
        "accepted_coverage",
        "selective_accuracy",
        "false_accept_rate_among_errors",
        "incorrect_high_priority_assignment_rate",
        "upward_priority_error_rate",
    )
    reference = {
        scenario["policy"]: scenario
        for scenario in scenarios
        if scenario["method"] == "uncalibrated"
    }

    def grouped_counts(scenario):
        masks = scenario["masks"]
        return {
            "rows": np.bincount(group_codes, minlength=group_count).astype(float),
            "accepted": np.bincount(
                group_codes,
                weights=masks["accepted"].astype(float),
                minlength=group_count,
            ),
            "correct_accepted": np.bincount(
                group_codes,
                weights=(masks["accepted"] & masks["correct"]).astype(float),
                minlength=group_count,
            ),
            "errors": np.bincount(
                group_codes,
                weights=masks["errors"].astype(float),
                minlength=group_count,
            ),
            "accepted_errors": np.bincount(
                group_codes,
                weights=(masks["accepted"] & masks["errors"]).astype(float),
                minlength=group_count,
            ),
            "unjustified_high": np.bincount(
                group_codes,
                weights=masks["unjustified_high"].astype(float),
                minlength=group_count,
            ),
            "upward": np.bincount(
                group_codes,
                weights=masks["upward"].astype(float),
                minlength=group_count,
            ),
        }

    def sampled_metrics(counts, selected):
        row_count = float(counts["rows"][selected].sum())
        accepted = float(counts["accepted"][selected].sum())
        errors = float(counts["errors"][selected].sum())
        accepted_errors = float(counts["accepted_errors"][selected].sum())
        return {
            "accepted_coverage": accepted / row_count if row_count else np.nan,
            "selective_accuracy": (
                float(counts["correct_accepted"][selected].sum()) / accepted
                if accepted else np.nan
            ),
            "false_accept_rate_among_errors": (
                accepted_errors / errors if errors else np.nan
            ),
            "incorrect_high_priority_assignment_rate": (
                float(counts["unjustified_high"][selected].sum()) / row_count
                if row_count else np.nan
            ),
            "upward_priority_error_rate": (
                float(counts["upward"][selected].sum()) / row_count
                if row_count else np.nan
            ),
        }

    reference_counts = {
        policy: grouped_counts(scenario)
        for policy, scenario in reference.items()
    }
    rows = []
    for scenario in scenarios:
        if scenario["method"] == "uncalibrated":
            continue
        baseline = reference.get(scenario["policy"])
        if baseline is None:
            raise ValueError(
                f"No uncalibrated reference for policy {scenario['policy']}."
            )
        candidate_counts = grouped_counts(scenario)
        baseline_counts = reference_counts[scenario["policy"]]
        deltas = {metric: [] for metric in metric_names}
        for _ in range(BOOTSTRAP_REPLICATES):
            selected = rng.integers(0, group_count, size=group_count)
            candidate_values = sampled_metrics(candidate_counts, selected)
            baseline_values = sampled_metrics(baseline_counts, selected)
            for metric in metric_names:
                deltas[metric].append(
                    candidate_values[metric] - baseline_values[metric]
                )
        for metric in metric_names:
            samples = np.asarray(deltas[metric], dtype=float)
            samples = samples[np.isfinite(samples)]
            point = (
                scenario["metrics"].get(metric)
                - baseline["metrics"].get(metric)
                if scenario["metrics"].get(metric) is not None
                and baseline["metrics"].get(metric) is not None
                else None
            )
            rows.append({
                "method": scenario["method"],
                "policy": scenario["policy"],
                "comparison_reference": "uncalibrated_same_policy",
                "metric": metric,
                "delta_candidate_minus_reference": point,
                "paired_ci_95_lower": (
                    float(np.quantile(samples, 0.025)) if len(samples) else None
                ),
                "paired_ci_95_upper": (
                    float(np.quantile(samples, 0.975)) if len(samples) else None
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": seed,
                "resampling_unit": "same exact-feature group draws",
            })
    return rows


def validate_qos_mapping(class_names):
    missing = sorted(set(class_names) - set(QOS_POLICY))
    if missing:
        raise ValueError(f"QoS mapping lacks classes: {missing}")
    medium = [
        label for label, mapping in QOS_POLICY.items()
        if mapping[0] == "MEDIUM" and mapping[3] == "AF21" and mapping[4] == 18
    ]
    if not medium or not any(QOS_POLICY[label][1] == 3 for label in medium):
        raise ValueError(
            "AF21 DSCP 18 is not assigned to logical Queue_Number 3."
        )
    if not QOS_COMPARE_PATH.is_file():
        raise FileNotFoundError(f"Mininet configuration not found: {QOS_COMPARE_PATH}")
    qos_source = QOS_COMPARE_PATH.read_text(encoding="utf-8")
    tos_filter = "match ip tos 0x48 0xfc flowid 1:20"
    htb_class = "classid 1:20 htb"
    if tos_filter not in qos_source or htb_class not in qos_source:
        raise ValueError(
            "Mininet source no longer maps AF21 ToS 0x48 to HTB class 1:20."
        )
    class_ids = {
        line.split("classid ", 1)[1].split(" ", 1)[0]
        for line in qos_source.splitlines()
        if "classid " in line and " htb " in line
    }
    if "1:3" in class_ids:
        physical_mapping = "AF21 ToS 0x48 -> HTB class 1:20; literal 1:3 exists"
    else:
        physical_mapping = "AF21 ToS 0x48 -> HTB class 1:20; no literal HTB class 1:3"
    return {
        "dscp_name": "AF21",
        "dscp_value": 18,
        "tos_match": "0x48",
        "logical_queue_number": 3,
        "physical_htb_classid": "1:20",
        "literal_htb_class_1_3_present": "1:3" in class_ids,
        "verification": physical_mapping,
        "source_file": str(QOS_COMPARE_PATH),
    }


def load_source_experiment(run_directory):
    run_directory = Path(run_directory).resolve()
    manifest_path = run_directory / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = (
        PROJECT_ROOT / "Scenario B-ARFF" / manifest["dataset"]["filename"]
    )
    mapping_path = run_directory / manifest["label_mapping_file"]
    indices_path = run_directory / manifest["split"]["index_file"]
    original_model_path = run_directory / manifest["model"]["artifact_file"]
    expected_hashes = (
        (dataset_path, manifest["dataset"]["sha256"], "dataset"),
        (
            mapping_path,
            manifest["label_mapping_sha256"],
            "label mapping",
        ),
        (original_model_path, manifest["model"]["sha256"], "baseline model"),
    )
    for path, expected, description in expected_hashes:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{description} SHA-256 mismatch: expected {expected}, got {actual}"
            )

    current_versions = installed_versions()
    mismatches = {
        key: (expected, current_versions.get(key))
        for key, expected in manifest["environment"].items()
        if current_versions.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Runtime versions differ from source run: {mismatches}")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    classes = mapping["classes"]
    if mapping["class_to_index"] != {
        label: index for index, label in enumerate(classes)
    }:
        raise ValueError("Saved label mapping is inconsistent.")
    if len(classes) != 14:
        raise ValueError(f"Expected 14 saved traffic classes, got {len(classes)}.")
    qos_verification = validate_qos_mapping(classes)

    with np.load(indices_path, allow_pickle=False) as saved_indices:
        train_indices = saved_indices["train_source_indices"].astype(np.int64)
        test_indices = saved_indices["test_source_indices"].astype(np.int64)
        clean_indices = saved_indices["clean_source_indices"].astype(np.int64)
    if sha256_indices(train_indices) != manifest["split"]["train_source_indices_sha256"]:
        raise ValueError("Saved training indices do not match the manifest.")
    if sha256_indices(test_indices) != manifest["split"]["test_source_indices_sha256"]:
        raise ValueError("Saved test indices do not match the manifest.")
    if len(test_indices) != 3622:
        raise ValueError(f"Expected 3,622 final test rows, got {len(test_indices)}.")
    if set(train_indices).intersection(test_indices):
        raise ValueError("Saved training and test indices overlap.")
    if set(train_indices).union(test_indices) != set(clean_indices):
        raise ValueError("Saved train/test indices do not partition clean rows.")

    dataframe = pd.read_csv(dataset_path)
    features = manifest["preprocessing"]["feature_columns"]
    target_column = manifest["preprocessing"]["target_column"]
    if dataframe.columns[:-1].tolist() != features:
        raise ValueError("Dataset feature columns/order differ from the manifest.")
    if dataframe.columns[-1] != target_column:
        raise ValueError("Dataset target column differs from the manifest.")
    metadata_names = {
        "capture_id",
        "capture",
        "session_id",
        "session",
        "pcap_id",
        "trace_id",
    }
    metadata_columns = [
        column for column in dataframe.columns
        if column.casefold() in metadata_names
    ]
    X_all = (
        dataframe.loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    y_strings = dataframe[target_column].astype(str)
    unknown = sorted(set(y_strings.iloc[clean_indices]) - set(classes))
    if unknown:
        raise ValueError(f"Dataset has labels absent from mapping: {unknown}")
    class_to_index = mapping["class_to_index"]
    y_all = y_strings.map(class_to_index).to_numpy(dtype=np.int64)
    return {
        "run_directory": run_directory,
        "manifest": manifest,
        "dataset_path": dataset_path,
        "mapping_path": mapping_path,
        "original_model_path": original_model_path,
        "classes": classes,
        "X_all": X_all,
        "y_all": y_all,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "clean_indices": clean_indices,
        "metadata_columns": metadata_columns,
        "qos_verification": qos_verification,
        "versions": current_versions,
    }


def split_training_indices(X_all, y_all, train_indices):
    X_train = X_all.iloc[train_indices]
    y_train = y_all[train_indices]
    groups = pd.util.hash_pandas_object(
        X_train,
        index=False,
    ).to_numpy(dtype=np.uint64)
    splitter = StratifiedGroupKFold(
        n_splits=SPLIT_COUNT,
        shuffle=True,
        random_state=SPLIT_SEED,
    )
    folds = []
    for _, validation_relative in splitter.split(X_train, y_train, groups):
        folds.append(train_indices[validation_relative])
    if len(folds) != SPLIT_COUNT:
        raise RuntimeError("Stratified group splitter returned an unexpected fold count.")
    partitions = {
        "model_training": np.sort(np.concatenate(
            [folds[index] for index in MODEL_FOLDS]
        )),
        "calibration_fitting": np.sort(folds[CALIBRATION_FOLD]),
        "policy_validation": np.sort(folds[POLICY_FOLD]),
    }
    partitions["final_test"] = np.sort(np.asarray([], dtype=np.int64))
    all_dev = np.concatenate([
        partitions["model_training"],
        partitions["calibration_fitting"],
        partitions["policy_validation"],
    ])
    if len(all_dev) != len(train_indices) or set(all_dev) != set(train_indices):
        raise ValueError("Phase 4 development partitions do not cover saved training rows.")
    for left_name, left in partitions.items():
        if left_name == "final_test":
            continue
        for right_name, right in partitions.items():
            if right_name <= left_name or right_name == "final_test":
                continue
            if set(left).intersection(right):
                raise ValueError(f"Partitions overlap: {left_name}, {right_name}.")
    feature_group_sets = {}
    for name in ("model_training", "calibration_fitting", "policy_validation"):
        feature_group_sets[name] = set(pd.util.hash_pandas_object(
            X_all.iloc[partitions[name]],
            index=False,
        ).to_numpy(dtype=np.uint64).tolist())
    for left_name, left_groups in feature_group_sets.items():
        for right_name, right_groups in feature_group_sets.items():
            if right_name > left_name and left_groups.intersection(right_groups):
                raise ValueError(
                    f"Exact-feature groups overlap {left_name} and {right_name}."
                )
    return partitions, feature_group_sets


def fit_temperature(base_model, X_calibration, y_calibration):
    imputer = base_model.named_steps["imputer"]
    classifier = base_model.named_steps["classifier"]
    transformed = imputer.transform(X_calibration)
    margins = np.asarray(classifier.predict(
        transformed,
        output_margin=True,
    ))
    if margins.ndim != 2 or margins.shape[1] != len(classifier.classes_):
        raise ValueError(f"Unexpected XGBoost margin shape: {margins.shape}")

    def objective(log_temperature):
        scaled = margins / np.exp(log_temperature)
        return float(np.mean(
            logsumexp(scaled, axis=1)
            - scaled[np.arange(len(y_calibration)), y_calibration]
        ))

    result = minimize_scalar(
        objective,
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    return TemperatureScaledModel(base_model, temperature), temperature, float(result.fun)


def write_report(
    path,
    manifest,
    partitions,
    metadata_columns,
    qos_verification,
    test_metrics,
):
    lines = [
        "Phase 4: Confidence-aware QoS decision experiment",
        "=" * 58,
        "",
        "Data boundaries",
        f"- Model-training rows: {len(partitions['model_training'])}.",
        f"- Calibration-fitting rows: {len(partitions['calibration_fitting'])}.",
        f"- Policy-validation rows: {len(partitions['policy_validation'])}.",
        f"- Preserved final test rows: {len(partitions['final_test'])}.",
        "- Calibration methods share one newly trained Phase 4 base XGBoost model.",
        "- Thresholds are fitted only from policy-validation probabilities and labels.",
        "- Final test data is used only for the frozen final evaluation.",
        "- Original model, original baseline, DSCP map, and prior experiments are unchanged.",
        "",
        "Grouping limitation",
        (
            f"- Capture/session metadata columns found: {metadata_columns}."
            if metadata_columns
            else "- No capture/session identifier exists in the source CSV; exact-feature hashes are the available grouping unit."
        ),
        "- All Phase 4 folds are disjoint by exact-feature group.",
        "",
        "Calibration interpretation",
        "- The Phase 3 test comparison is exploratory and did not identify one method that improves all probability metrics.",
        "- Phase 4 includes uncalibrated, sigmoid, isotonic, and temperature probabilities using the same base model and data partitions.",
        "- Probability quality and QoS decision quality are reported separately; calibration does not imply better decisions.",
        "",
        "Uncertain-flow fallback and HTB mapping",
        f"- Logical fallback is AF21 / DSCP 18 / Queue_Number 3; verification: {qos_verification['verification']}.",
        "- The current Mininet setup has HTB class 1:20 for AF21, not a literal HTB class 1:3.",
        "- The fallback comparison therefore distinguishes the dataset's logical queue number from the physical Mininet class ID.",
        "- No DSCP mapping values were changed.",
        "",
        "Final-test policy results",
        "Method | Policy | Coverage | Selective accuracy | Macro-F1 | False acceptance among errors | Incorrect high assignment | Upward priority error",
        "-" * 137,
    ]
    for row in test_metrics:
        lines.append(
            f"{row['method']} | {row['policy']} | "
            f"{row['accepted_coverage']:.4f} | "
            f"{row['selective_accuracy'] if row['selective_accuracy'] is not None else float('nan'):.4f} | "
            f"{row['macro_f1']:.4f} | "
            f"{row['false_accept_rate_among_errors'] if row['false_accept_rate_among_errors'] is not None else float('nan'):.4f} | "
            f"{row['incorrect_high_priority_assignment_rate']:.4f} | "
            f"{row['upward_priority_error_rate']:.4f}"
        )
    lines.extend([
        "",
        "Interpretation limits",
        "- The policy-validation partition, not the final test, determined confidence thresholds.",
        "- Policy-validation threshold statistics are selection estimates and may be optimistic.",
        "- The table is comparative evidence, not a selected operational winner.",
        "- This experiment is offline. It does not demonstrate improved latency, loss, throughput, fairness, or any other network QoS outcome.",
        "- Network QoS claims require separate controlled Mininet/HTB or real-network experiments with repeated traffic runs.",
        f"- Source baseline accuracy for context: {manifest['metrics']['accuracy']:.4%}.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(run_directory):
    source = load_source_experiment(run_directory)
    manifest = source["manifest"]
    classes = source["classes"]
    X_all = source["X_all"]
    y_all = source["y_all"]
    train_indices = source["train_indices"]
    test_indices = source["test_indices"]
    partitions, feature_group_sets = split_training_indices(
        X_all,
        y_all,
        train_indices,
    )
    partitions["final_test"] = test_indices.copy()
    for name, indices in partitions.items():
        missing = sorted(set(range(len(classes))) - set(y_all[indices].tolist()))
        if missing:
            raise ValueError(
                f"{name} split lacks classes: {[classes[index] for index in missing]}"
            )
    test_groups = set(pd.util.hash_pandas_object(
        X_all.iloc[test_indices],
        index=False,
    ).to_numpy(dtype=np.uint64).tolist())
    for partition_name, groups in feature_group_sets.items():
        if groups.intersection(test_groups):
            raise ValueError(
                f"Exact-feature groups overlap {partition_name} and final test."
            )

    model_parameters = dict(manifest["model"]["xgboost_parameters"])
    if model_parameters.get("num_class") != len(classes):
        raise ValueError("Saved XGBoost class count differs from the label mapping.")
    experiment_directory = create_experiment_directory(source["run_directory"])

    X_model = X_all.iloc[partitions["model_training"]]
    y_model = y_all[partitions["model_training"]]
    X_calibration = X_all.iloc[partitions["calibration_fitting"]]
    y_calibration = y_all[partitions["calibration_fitting"]]
    X_policy = X_all.iloc[partitions["policy_validation"]]
    y_policy = y_all[partitions["policy_validation"]]
    X_test = X_all.iloc[test_indices]
    y_test = y_all[test_indices]

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(**model_parameters)),
    ])
    fit_started = time.time()
    base_model.fit(X_model, y_model)
    fit_seconds = float(time.time() - fit_started)
    classifier = base_model.named_steps["classifier"]
    if not np.array_equal(classifier.classes_, np.arange(len(classes))):
        raise ValueError("Phase 4 model class order differs from saved mapping.")

    sigmoid_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid",
        ensemble="auto",
    )
    sigmoid_model.fit(X_calibration, y_calibration)
    isotonic_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="isotonic",
        ensemble="auto",
    )
    isotonic_model.fit(X_calibration, y_calibration)
    temperature_model, temperature, temperature_nll = fit_temperature(
        base_model,
        X_calibration,
        y_calibration,
    )
    models = {
        "uncalibrated": base_model,
        "sigmoid": sigmoid_model,
        "isotonic": isotonic_model,
        "temperature": temperature_model,
    }
    for name, model in models.items():
        joblib.dump(model, experiment_directory / f"{name}_model.joblib")

    policy_probabilities = {
        name: normalize_probabilities(model.predict_proba(X_policy))
        for name, model in models.items()
    }
    test_probabilities = {
        name: normalize_probabilities(model.predict_proba(X_test))
        for name, model in models.items()
    }
    expected_shape = (len(test_indices), len(classes))
    for method, probabilities in test_probabilities.items():
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"{method} returned probability shape {probabilities.shape}, "
                f"expected {expected_shape}."
            )
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{method} returned non-finite test probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{method} probabilities do not sum to one.")

    thresholds = {}
    threshold_rows = []
    for method, probabilities in policy_probabilities.items():
        method_thresholds = build_thresholds(
            method,
            probabilities,
            y_policy,
            classes,
        )
        thresholds[method] = method_thresholds
        for target_key, values in method_thresholds.items():
            for scope in ("global", "class"):
                threshold_rows.append({
                    "method": method,
                    "threshold_scope": scope,
                    "target_coverage": values["target_coverage"],
                    "global_threshold": values["global_threshold"],
                    "policy_validation_accepted_rows": (
                        values["global_policy_accepted_rows"]
                        if scope == "global"
                        else values["class_policy_accepted_rows"]
                    ),
                    "policy_validation_rows": len(y_policy),
                    "policy_validation_achieved_coverage": (
                        values["global_policy_achieved_coverage"]
                        if scope == "global"
                        else values["class_policy_achieved_coverage"]
                    ),
                    "minimum_predicted_class_support": MIN_CLASS_POLICY_SUPPORT,
                    "class_threshold_fallback_classes": ",".join(
                        values["class_threshold_fallback_classes"]
                    ),
                    "class_thresholds_json": json.dumps(
                        values["class_thresholds"],
                        sort_keys=True,
                    ),
                })

    test_metrics = []
    class_rows = []
    decision_rows = []
    scenarios = []
    for method, probabilities in test_probabilities.items():
        predicted = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        method_decisions = []
        normal_assigned_labels = [classes[index] for index in predicted]
        normal_metrics, normal_masks = policy_metrics(
            y_test,
            predicted,
            confidence,
            np.ones(len(y_test), dtype=bool),
            normal_assigned_labels,
            classes,
            probabilities,
        )
        normal_record = {
            "method": method,
            "policy": "normal_mapping",
            "threshold_scope": "none",
            "target_coverage": 1.0,
            "selected_threshold": None,
            "fallback": "none",
            **normal_metrics,
        }
        test_metrics.append(normal_record)
        scenarios.append({
            "method": method,
            "policy": "normal_mapping",
            "metrics": normal_metrics,
            "masks": normal_masks,
        })
        class_rows.extend(class_reliability_rows(
            method,
            "normal_mapping",
            y_test,
            predicted,
            confidence,
            np.ones(len(y_test), dtype=bool),
            probabilities,
            np.asarray([
                PRIORITY_RANK[QOS_POLICY[label][0]]
                for label in normal_assigned_labels
            ]),
            classes,
        ))
        method_decisions.append({
            "policy": "normal_mapping",
            "scope": "none",
            "target": 1.0,
            "accepted": np.ones(len(y_test), dtype=bool),
            "thresholds": None,
        })

        for target_key, threshold_configuration in thresholds[method].items():
            for scope in ("global", "class"):
                if scope == "global":
                    selected_thresholds = np.full(
                        len(y_test),
                        threshold_configuration["global_threshold"],
                    )
                else:
                    selected_thresholds = np.asarray([
                        threshold_configuration["class_thresholds"][
                            classes[index]
                        ]
                        for index in predicted
                    ])
                accepted = confidence >= selected_thresholds
                fallback_actions = {
                    "neutral_af21_logical_queue_3": "neutral",
                    "cap_high_to_af21": "cap_high",
                }
                for fallback_name, fallback_kind in fallback_actions.items():
                    assigned_labels = normal_assigned_labels.copy()
                    rejected = ~accepted
                    if fallback_kind == "neutral":
                        assigned_labels = [
                            "BROWSING" if rejected[index] else label
                            for index, label in enumerate(assigned_labels)
                        ]
                    else:
                        assigned_labels = [
                            (
                                "BROWSING"
                                if rejected[index]
                                and PRIORITY_RANK[QOS_POLICY[label][0]] > 2
                                else label
                            )
                            for index, label in enumerate(assigned_labels)
                        ]
                    policy_name = (
                        f"{scope}_{target_key}_{fallback_name}"
                    )
                    metrics, masks = policy_metrics(
                        y_test,
                        predicted,
                        confidence,
                        accepted,
                        assigned_labels,
                        classes,
                        probabilities,
                    )
                    record = {
                        "method": method,
                        "policy": policy_name,
                        "threshold_scope": scope,
                        "target_coverage": float(target_key),
                        "selected_threshold": (
                            threshold_configuration["global_threshold"]
                            if scope == "global" else None
                        ),
                        "fallback": fallback_name,
                        **metrics,
                    }
                    test_metrics.append(record)
                    scenarios.append({
                        "method": method,
                        "policy": policy_name,
                        "metrics": metrics,
                        "masks": masks,
                    })
                    assigned_priority = np.asarray([
                        PRIORITY_RANK[QOS_POLICY[label][0]]
                        for label in assigned_labels
                    ])
                    class_rows.extend(class_reliability_rows(
                        method,
                        policy_name,
                        y_test,
                        predicted,
                        confidence,
                        accepted,
                        probabilities,
                        assigned_priority,
                        classes,
                    ))
                    method_decisions.append({
                        "policy": policy_name,
                        "scope": scope,
                        "target": float(target_key),
                        "accepted": accepted,
                        "thresholds": selected_thresholds,
                        "assigned_labels": assigned_labels,
                    })

        for decision in method_decisions:
            accepted = decision["accepted"]
            assigned_labels = decision.get(
                "assigned_labels",
                normal_assigned_labels,
            )
            thresholds_for_rows = decision["thresholds"]
            for local_index, source_index in enumerate(test_indices):
                decision_rows.append({
                    "source_row_index": int(source_index),
                    "method": method,
                    "policy": decision["policy"],
                    "threshold_scope": decision["scope"],
                    "target_coverage": decision["target"],
                    "threshold": (
                        float(thresholds_for_rows[local_index])
                        if thresholds_for_rows is not None else None
                    ),
                    "actual_traffic_class": classes[y_test[local_index]],
                    "predicted_traffic_class": classes[predicted[local_index]],
                    "prediction_confidence": float(confidence[local_index]),
                    "accepted": bool(accepted[local_index]),
                    "prediction_correct": bool(
                        predicted[local_index] == y_test[local_index]
                    ),
                    "assigned_traffic_class": assigned_labels[local_index],
                    "assigned_priority": QOS_POLICY[assigned_labels[local_index]][0],
                    "assigned_queue_number": QOS_POLICY[assigned_labels[local_index]][1],
                    "assigned_dscp_name": QOS_POLICY[assigned_labels[local_index]][3],
                    "assigned_dscp_value": QOS_POLICY[assigned_labels[local_index]][4],
                })

    group_codes = pd.util.hash_pandas_object(
        X_test,
        index=False,
    ).to_numpy(dtype=np.uint64)
    _, group_codes = np.unique(group_codes, return_inverse=True)
    bootstrap_rows = cluster_bootstrap_intervals(
        scenarios,
        group_codes.astype(np.int64),
        int(group_codes.max()) + 1,
        BOOTSTRAP_SEED,
    )
    paired_bootstrap_rows = paired_cluster_bootstrap_deltas(
        scenarios,
        group_codes.astype(np.int64),
        int(group_codes.max()) + 1,
        BOOTSTRAP_SEED,
    )

    model_artifacts = {
        f"{name}_model.joblib": sha256_file(
            experiment_directory / f"{name}_model.joblib"
        )
        for name in models
    }
    np.savez_compressed(
        experiment_directory / "phase4_split_indices.npz",
        model_training_source_indices=partitions["model_training"],
        calibration_fitting_source_indices=partitions["calibration_fitting"],
        policy_validation_source_indices=partitions["policy_validation"],
        saved_test_source_indices=partitions["final_test"],
    )
    split_hashes = {
        name: sha256_indices(indices)
        for name, indices in partitions.items()
    }
    write_json(
        experiment_directory / "threshold_selection.json",
        {
            "selection_partition": "policy_validation",
            "thresholds": thresholds,
            "target_coverages": TARGET_COVERAGES,
            "minimum_predicted_class_support": MIN_CLASS_POLICY_SUPPORT,
            "class_specific_threshold_fallback": (
                "Use the same method's global threshold for predicted classes "
                "with fewer than the minimum support."
            ),
            "fallback_policies": {
                "neutral_af21_logical_queue_3": {
                    "dscp_name": "AF21",
                    "dscp_value": 18,
                    "logical_queue_number": 3,
                    "physical_htb_classid": "1:20",
                },
                "cap_high_to_af21": (
                    "For rejected predictions mapped HIGH, assign the existing "
                    "AF21 medium mapping; rejected medium/low predictions retain "
                    "their predicted-class mapping."
                ),
            },
        },
    )
    pd.DataFrame(threshold_rows).to_csv(
        experiment_directory / "policy_threshold_selection.csv",
        index=False,
        float_format="%.10g",
    )
    metrics_frame = pd.DataFrame(test_metrics)
    metrics_frame.to_csv(
        experiment_directory / "test_policy_evaluation.csv",
        index=False,
        float_format="%.10g",
    )
    pd.DataFrame(class_rows).to_csv(
        experiment_directory / "per_class_policy_reliability.csv",
        index=False,
        float_format="%.10g",
    )
    pd.DataFrame(decision_rows).to_csv(
        experiment_directory / "test_policy_decisions.csv",
        index=False,
        float_format="%.10g",
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        experiment_directory / "test_group_bootstrap_intervals.csv",
        index=False,
        float_format="%.10g",
    )
    pd.DataFrame(paired_bootstrap_rows).to_csv(
        experiment_directory / "paired_test_policy_deltas.csv",
        index=False,
        float_format="%.10g",
    )

    calibration_metrics = []
    for method, probabilities in policy_probabilities.items():
        calibration_metrics.append({
            "method": method,
            "partition": "policy_validation",
            **probability_metrics(y_policy, probabilities),
        })
    for method, probabilities in test_probabilities.items():
        calibration_metrics.append({
            "method": method,
            "partition": "final_test",
            **probability_metrics(y_test, probabilities),
        })
    pd.DataFrame(calibration_metrics).to_csv(
        experiment_directory / "probability_quality.csv",
        index=False,
        float_format="%.10g",
    )

    split_path = experiment_directory / "phase4_split_indices.npz"
    source_indices_hashes = {
        "dataset_sha256": sha256_file(source["dataset_path"]),
        "source_model_sha256": sha256_file(source["original_model_path"]),
        "label_mapping_sha256": sha256_file(source["mapping_path"]),
        "source_train_indices_sha256": sha256_indices(train_indices),
        "source_test_indices_sha256": sha256_indices(test_indices),
    }
    configuration = {
        "experiment_id": experiment_directory.name,
        "source_run_id": manifest["run_id"],
        "source_hashes": source_indices_hashes,
        "environment": source["versions"],
        "random_seeds": {
            "split": SPLIT_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "xgboost": model_parameters["random_state"],
        },
        "partitioning": {
            "method": "StratifiedGroupKFold",
            "n_splits": SPLIT_COUNT,
            "shuffle": True,
            "model_folds": MODEL_FOLDS,
            "calibration_fitting_fold": CALIBRATION_FOLD,
            "policy_validation_fold": POLICY_FOLD,
            "grouping": "exact-feature hash",
            "capture_session_metadata_columns": source["metadata_columns"],
            "capture_session_metadata_available": bool(source["metadata_columns"]),
            "saved_final_test_preserved": True,
        },
        "row_counts": {
            name: int(len(indices)) for name, indices in partitions.items()
        },
        "partition_indices_sha256": split_hashes,
        "xgboost_parameters": model_parameters,
        "calibration_methods": [
            "uncalibrated",
            "sigmoid",
            "isotonic",
            "temperature",
        ],
        "temperature_scaling": {
            "value": temperature,
            "calibration_log_loss": temperature_nll,
            "optimization": "bounded scalar minimization of calibration NLL over log(T) in [-5, 5]",
        },
        "threshold_selection": {
            "partition": "policy_validation",
            "target_coverages": TARGET_COVERAGES,
            "threshold_scopes": ["global", "predicted_class"],
            "minimum_class_support": MIN_CLASS_POLICY_SUPPORT,
            "test_used_for_threshold_selection": False,
        },
        "fallback_verification": source["qos_verification"],
        "model_fit_seconds": fit_seconds,
        "model_artifact_hashes": model_artifacts,
        "outputs": [
            "uncalibrated_model.joblib",
            "sigmoid_model.joblib",
            "isotonic_model.joblib",
            "temperature_model.joblib",
            "phase4_split_indices.npz",
            "threshold_selection.json",
            "policy_threshold_selection.csv",
            "test_policy_evaluation.csv",
            "per_class_policy_reliability.csv",
            "test_policy_decisions.csv",
            "test_group_bootstrap_intervals.csv",
            "paired_test_policy_deltas.csv",
            "probability_quality.csv",
            "phase4_configuration.json",
            "phase4_report.txt",
            "phase4_manifest.json",
        ],
        "network_qos_experiment_performed": False,
        "baseline_model_modified": False,
        "source_model_sha256_after_run": sha256_file(source["original_model_path"]),
    }
    write_json(experiment_directory / "phase4_configuration.json", configuration)
    write_report(
        experiment_directory / "phase4_report.txt",
        manifest,
        partitions,
        source["metadata_columns"],
        source["qos_verification"],
        test_metrics,
    )
    manifest_out = {
        "schema_version": 1,
        **configuration,
        "split_indices_file_sha256": sha256_file(split_path),
        "source_manifest_sha256": sha256_file(
            source["run_directory"] / "experiment_manifest.json"
        ),
    }
    write_json(experiment_directory / "phase4_manifest.json", manifest_out)

    if sha256_file(source["original_model_path"]) != manifest["model"]["sha256"]:
        raise RuntimeError("Original baseline model changed during Phase 4.")
    if sha256_file(source["dataset_path"]) != manifest["dataset"]["sha256"]:
        raise RuntimeError("Source dataset changed during Phase 4.")

    print("Phase 4 experiment:", experiment_directory)
    print("Partition rows:", configuration["row_counts"])
    print("Capture/session identifiers found:", source["metadata_columns"])
    print("AF21 fallback mapping:", source["qos_verification"]["verification"])
    print("Test rows preserved:", len(test_indices))
    print("Original baseline SHA-256 preserved:", source_indices_hashes["source_model_sha256"])
    print("Test policy records:", len(test_metrics))
    print("No network QoS claim: Mininet/HTB evaluation not performed.")
    print("Outputs:", ", ".join(configuration["outputs"]))
    return experiment_directory


def main():
    arguments = parse_args()
    run_experiment(arguments.run_dir)


if __name__ == "__main__":
    main()
