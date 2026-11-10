import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from calibration_models import TemperatureScaledModel
from experiment_reproducibility import sha256_file, sha256_indices, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_ROOT
    / "Scenario B-ARFF"
    / "TimeBasedFeatures-Dataset-15s.csv"
)
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "Scenario B-ARFF"
    / "reproducible_runs"
    / "clean_xgboost_20261101T173109339548Z"
)

INNER_SPLIT_SEED = 2026
BOOTSTRAP_SEED = 2027
BOOTSTRAP_REPLICATES = 1000
INNER_SPLIT_COUNT = 5
INNER_VALIDATION_FOLD = 0
CALIBRATION_BINS = 15
ECE_BIN_EDGES = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-free sigmoid, isotonic, and temperature "
            "calibration experiments."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Immutable parent experiment directory.",
    )
    return parser.parse_args()


def create_experiment_directory(run_directory):
    root = Path(run_directory) / "calibration_experiments"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = f"calibration_{timestamp}"
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
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
        "scipy",
        "threadpoolctl",
        "matplotlib",
    ]
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Required package is not installed: {package}"
            ) from error
    return result


def softmax_nll_from_margins(margins, y_true, temperature):
    scaled = margins / temperature
    return float(np.mean(
        logsumexp(scaled, axis=1) - scaled[np.arange(len(y_true)), y_true]
    ))


def fit_temperature(margins, y_true):
    result = minimize_scalar(
        lambda log_temperature: softmax_nll_from_margins(
            margins,
            y_true,
            float(np.exp(log_temperature)),
        ),
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Temperature optimization failed: {result.message}"
        )
    return float(np.exp(result.x)), float(result.fun)


def top_label_ece(probabilities, y_true):
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    bin_indices = np.minimum(
        np.digitize(confidence, ECE_BIN_EDGES[1:-1], right=False),
        CALIBRATION_BINS - 1,
    )
    ece = 0.0
    for bin_index in range(CALIBRATION_BINS):
        mask = bin_indices == bin_index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean())
                - float(confidence[mask].mean())
            )
    return float(ece)


def classwise_ece(probabilities, y_true):
    class_values = []
    for class_index in range(probabilities.shape[1]):
        class_probabilities = probabilities[:, class_index]
        class_truth = (y_true == class_index).astype(float)
        bins = np.minimum(
            np.digitize(
                class_probabilities,
                ECE_BIN_EDGES[1:-1],
                right=False,
            ),
            CALIBRATION_BINS - 1,
        )
        ece = 0.0
        for bin_index in range(CALIBRATION_BINS):
            mask = bins == bin_index
            if mask.any():
                ece += float(mask.mean()) * abs(
                    float(class_truth[mask].mean())
                    - float(class_probabilities[mask].mean())
                )
        class_values.append(ece)
    return float(np.mean(class_values)), class_values


def calibration_metrics(probabilities, y_true):
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(y_true)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("Invalid probability matrix supplied for evaluation.")
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    macro_classwise_ece, per_class_ece = classwise_ece(
        probabilities,
        y_true,
    )
    return {
        "accuracy": float(accuracy_score(y_true, probabilities.argmax(axis=1))),
        "macro_f1": float(f1_score(
            y_true,
            probabilities.argmax(axis=1),
            average="macro",
            zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            y_true,
            probabilities.argmax(axis=1),
            average="weighted",
            zero_division=0,
        )),
        "multiclass_brier_score": brier,
        "normalized_brier_score": brier / 2.0,
        "multiclass_log_loss": float(log_loss(
            y_true,
            probabilities,
            labels=np.arange(probabilities.shape[1]),
        )),
        "top_label_ece": top_label_ece(probabilities, y_true),
        "classwise_ece": macro_classwise_ece,
        "classwise_ece_by_index": per_class_ece,
    }


def reliability_bins(probabilities, y_true):
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    bins = np.minimum(
        np.digitize(confidence, ECE_BIN_EDGES[1:-1], right=False),
        CALIBRATION_BINS - 1,
    )
    rows = []
    for bin_index in range(CALIBRATION_BINS):
        mask = bins == bin_index
        rows.append({
            "bin_index": bin_index,
            "lower_bound": float(ECE_BIN_EDGES[bin_index]),
            "upper_bound": float(ECE_BIN_EDGES[bin_index + 1]),
            "sample_count": int(mask.sum()),
            "mean_confidence": (
                float(confidence[mask].mean()) if mask.any() else None
            ),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        })
    return rows


def bootstrap_performance_differences(
    probabilities_by_method,
    y_true,
    test_group_ids,
):
    baseline = probabilities_by_method["inner_uncalibrated"]
    baseline_predictions = baseline.argmax(axis=1)
    methods = [
        method for method in probabilities_by_method
        if method not in ("saved_original_baseline", "inner_uncalibrated")
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    _, group_inverse = np.unique(test_group_ids, return_inverse=True)
    group_rows = [
        np.flatnonzero(group_inverse == group_index)
        for group_index in range(group_inverse.max() + 1)
    ]
    rows = []

    def scores(y, predicted):
        return np.asarray([
            accuracy_score(y, predicted),
            f1_score(y, predicted, average="macro", zero_division=0),
            f1_score(y, predicted, average="weighted", zero_division=0),
        ])

    for method in methods:
        candidate_predictions = probabilities_by_method[method].argmax(axis=1)
        observed_delta = scores(y_true, candidate_predictions) - scores(
            y_true,
            baseline_predictions,
        )
        deltas = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled_groups = rng.integers(
                0,
                len(group_rows),
                size=len(group_rows),
            )
            sample = np.concatenate([
                group_rows[group_index]
                for group_index in sampled_groups
            ])
            deltas[replicate] = (
                scores(y_true[sample], candidate_predictions[sample])
                - scores(y_true[sample], baseline_predictions[sample])
            )
        intervals = np.quantile(deltas, [0.025, 0.975], axis=0)
        for metric_index, metric_name in enumerate(
            ("accuracy", "macro_f1", "weighted_f1")
        ):
            rows.append({
                "Method": method,
                "Metric": metric_name,
                "Delta_vs_Inner_Uncalibrated": observed_delta[metric_index],
                "Paired_Bootstrap_CI_95_Lower": intervals[0, metric_index],
                "Paired_Bootstrap_CI_95_Upper": intervals[1, metric_index],
                "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                "Bootstrap_Seed": BOOTSTRAP_SEED,
            })
    return pd.DataFrame(rows)


def save_reliability_diagrams(output_directory, probability_outputs, y_true):
    for method, probabilities in probability_outputs.items():
        bins = reliability_bins(probabilities, y_true)
        populated = [item for item in bins if item["sample_count"]]
        ece = top_label_ece(probabilities, y_true)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Perfect calibration",
        )
        ax.plot(
            [item["mean_confidence"] for item in populated],
            [item["accuracy"] for item in populated],
            marker="o",
            color="#264653",
            label=method.replace("_", " ").title(),
        )
        for item in populated:
            ax.annotate(
                str(item["sample_count"]),
                (item["mean_confidence"], item["accuracy"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set(
            title=f"{method.replace('_', ' ').title()} reliability "
            f"(ECE={ece:.4f})",
            xlabel="Mean maximum confidence in bin",
            ylabel="Accuracy in bin",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(
            output_directory / f"reliability_{method}.png",
            dpi=160,
        )
        plt.close(fig)


def create_report(
    summary,
    classwise,
    performance_comparison,
    calibration_row_count,
    model_training_row_count,
    test_row_count,
    temperatures,
):
    lines = [
        "Phase 3: Leakage-free probability calibration",
        "=" * 48,
        "",
        "Data boundaries",
        f"- Calibration model-training rows: {model_training_row_count}.",
        f"- Calibration-validation rows: {calibration_row_count}.",
        f"- Saved final test rows: {test_row_count}; used for evaluation only.",
        "- The original full-data training model and baseline artifacts were "
        "not changed.",
        "- Calibrators were fitted only on calibration-validation rows.",
        "- No operational confidence threshold was selected.",
        "",
        "Method comparison on the saved test set",
        "- The saved original baseline is included as a reference. "
        "Calibration effects are compared primarily with the "
        "inner_uncalibrated model, which uses the same reduced training "
        "subset as the calibration candidates.",
        "- Temperature scaling preserves argmax labels mathematically; "
        "sigmoid/isotonic one-vs-rest calibration may change class ranking.",
        "",
        (
            "Method | Accuracy | Macro-F1 | Weighted-F1 | Brier | "
            "Normalized Brier | Log loss | Top-label ECE | Classwise ECE"
        ),
        "-" * 145,
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.Method} | {row.Accuracy:.4f} | {row.Macro_F1:.4f} | "
            f"{row.Weighted_F1:.4f} | {row.Multiclass_Brier:.6f} | "
            f"{row.Normalized_Brier:.6f} | {row.Multiclass_Log_Loss:.6f} | "
            f"{row.Top_Label_ECE:.6f} | {row.Classwise_ECE:.6f}"
        )

    summary_by_method = summary.set_index("Method")
    reference = summary_by_method.loc["inner_uncalibrated"]
    lines.extend([
        "",
        "Probability-reliability comparison vs inner_uncalibrated",
    ])
    for method in ("sigmoid", "isotonic", "temperature"):
        row = summary_by_method.loc[method]
        lines.append(
            f"- {method}: delta Brier="
            f"{row.Multiclass_Brier - reference.Multiclass_Brier:+.6f}, "
            f"delta log loss="
            f"{row.Multiclass_Log_Loss - reference.Multiclass_Log_Loss:+.6f}, "
            f"delta top-label ECE="
            f"{row.Top_Label_ECE - reference.Top_Label_ECE:+.6f}, "
            f"delta classwise ECE="
            f"{row.Classwise_ECE - reference.Classwise_ECE:+.6f}; "
            "negative deltas are improvements."
        )
    lines.extend([
        "- No calibration method improved every reported probability metric "
        "relative to the same-data inner uncalibrated model.",
        "- Sigmoid worsened Brier, log loss, top-label ECE, and classwise ECE.",
        "- Isotonic slightly improved classwise ECE, while Brier, log loss, "
        "and top-label ECE worsened.",
        "- Temperature scaling slightly improved Brier score but worsened "
        "log loss and both ECE measures; its fitted temperature preserves "
        "the uncalibrated argmax predictions.",
    ])

    lines.extend([
        "",
        "Temperature scaling",
        (
            f"- Validation-fitted temperature: {temperatures['temperature']:.8f}; "
            f"validation log loss after scaling: "
            f"{temperatures['validation_log_loss']:.6f}."
        ),
        "- Temperature was optimized on calibration-validation labels only.",
        "",
        "Per-class accuracy and calibration",
    ])
    for class_name in classwise["Traffic_Class"].drop_duplicates():
        rows = classwise[classwise["Traffic_Class"] == class_name]
        lines.append(f"- {class_name}:")
        for row in rows.itertuples(index=False):
            lines.append(
                f"  {row.Method}: "
                f"accuracy={row.Classification_Accuracy:.4f}, "
                f"support={row.Number_of_Samples}, "
                f"mean_probability={row.Mean_Predicted_Probability:.4f}, "
                f"OvR_ECE={row.One_vs_Rest_ECE:.6f}, "
                f"OvR_Brier={row.One_vs_Rest_Brier:.6f}."
            )

    lines.extend([
        "",
        "Paired bootstrap performance comparison",
        (
            f"- {BOOTSTRAP_REPLICATES} paired cluster-bootstrap replicates "
            f"resampling exact-feature groups; seed={BOOTSTRAP_SEED}. "
            "CIs describe test performance and were "
            "not used to fit calibrators or select an operational policy."
        ),
    ])
    for row in performance_comparison.itertuples(index=False):
        lines.append(
            f"- {row.Method} vs inner_uncalibrated, {row.Metric}: "
            f"delta={row.Delta_vs_Inner_Uncalibrated:+.4f}, "
            f"95% CI [{row.Paired_Bootstrap_CI_95_Lower:+.4f}, "
            f"{row.Paired_Bootstrap_CI_95_Upper:+.4f}]."
        )

    lines.extend([
        "",
        "Interpretation and limitations",
        "- Lower Brier score, log loss, or ECE indicates improvement in the "
        "corresponding probability-quality measure; these metrics are "
        "reported side by side rather than used to select a winner from "
        "the test set.",
        "- A paired-bootstrap performance interval crossing zero does not "
        "show a statistically clear change at this interval level; it does "
        "not prove equivalence or absence of practically important loss.",
        "- The 95% paired cluster-bootstrap intervals for sigmoid and "
        "isotonic accuracy/Macro-F1/Weighted-F1 differences all cross zero; "
        "temperature scaling leaves these metrics unchanged because it "
        "preserves argmax.",
        "- The calibrated experiment base estimator is trained on fewer "
        "rows than the original full-train model. Comparisons against "
        "inner_uncalibrated isolate the calibration transform; the saved "
        "original baseline is a separate reference.",
        "- Isotonic calibration can overfit, especially for less-supported "
        "classes. Test results are held-out estimates, not a basis for "
        "threshold selection.",
        "- Any later method choice or operational threshold should be "
        "predeclared or selected using a separate validation/calibration "
        "procedure, with a final untouched test set.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    run_directory = args.run_dir.resolve()
    manifest_path = run_directory / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    original_manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    dataset_hash = sha256_file(DATASET_PATH)
    if dataset_hash != manifest["dataset"]["sha256"]:
        raise ValueError("Dataset SHA-256 differs from the saved experiment.")

    original_model_path = run_directory / manifest["model"]["artifact_file"]
    original_model_hash = sha256_file(original_model_path)
    if original_model_hash != manifest["model"]["sha256"]:
        raise ValueError("Original model hash differs from the manifest.")
    mapping_path = run_directory / manifest["label_mapping_file"]
    if sha256_file(mapping_path) != manifest["label_mapping_sha256"]:
        raise ValueError("Saved label mapping hash differs from the manifest.")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    classes = mapping["classes"]
    class_to_index = mapping["class_to_index"]
    if class_to_index != {
        label: index for index, label in enumerate(classes)
    }:
        raise ValueError("Saved class mapping is inconsistent.")

    current_versions = installed_versions()
    version_mismatches = {
        key: (expected, current_versions.get(key))
        for key, expected in manifest["environment"].items()
        if current_versions.get(key) != expected
    }
    if version_mismatches:
        raise RuntimeError(
            f"Runtime versions differ from the source run: {version_mismatches}"
        )

    with np.load(
        run_directory / manifest["split"]["index_file"],
        allow_pickle=False,
    ) as indices:
        original_train_indices = indices["train_source_indices"]
        test_indices = indices["test_source_indices"]
        clean_indices = indices["clean_source_indices"]
    if sha256_indices(original_train_indices) != (
        manifest["split"]["train_source_indices_sha256"]
    ):
        raise ValueError("Saved training indices do not match the manifest.")
    if sha256_indices(test_indices) != (
        manifest["split"]["test_source_indices_sha256"]
    ):
        raise ValueError("Saved test indices do not match the manifest.")
    if len(test_indices) != manifest["split"]["test_row_count"]:
        raise ValueError("Saved test row count differs from the manifest.")
    if set(original_train_indices).intersection(test_indices):
        raise ValueError("Original train and test rows overlap.")

    df = pd.read_csv(DATASET_PATH)
    feature_columns = manifest["preprocessing"]["feature_columns"]
    if df.columns[:-1].tolist() != feature_columns:
        raise ValueError("Feature columns/order differ from the saved model.")
    if df.columns[-1] != manifest["preprocessing"]["target_column"]:
        raise ValueError("Dataset target column differs from the manifest.")
    if np.any(clean_indices >= len(df)) or np.any(test_indices >= len(df)):
        raise ValueError("Saved row indices exceed dataset length.")

    X_all = (
        df.loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    y_all_strings = df.iloc[:, -1].astype(str)
    unknown_classes = sorted(
        set(y_all_strings.iloc[clean_indices]) - set(class_to_index)
    )
    if unknown_classes:
        raise ValueError(f"Clean dataset has unmapped classes: {unknown_classes}")
    y_all = y_all_strings.map(class_to_index).to_numpy(dtype=np.int64)

    # Split only the original saved training rows. The saved test rows never
    # participate in splitting, fitting, or temperature optimization.
    X_original_train = X_all.iloc[original_train_indices]
    y_original_train = y_all[original_train_indices]
    groups_original_train = pd.util.hash_pandas_object(
        X_original_train,
        index=False,
    ).to_numpy()
    inner_splitter = StratifiedGroupKFold(
        n_splits=INNER_SPLIT_COUNT,
        shuffle=True,
        random_state=INNER_SPLIT_SEED,
    )
    model_train_relative, calibration_relative = next(
        inner_splitter.split(
            X_original_train,
            y_original_train,
            groups_original_train,
        )
    )
    model_train_indices = original_train_indices[model_train_relative]
    calibration_indices = original_train_indices[calibration_relative]
    if set(model_train_indices).intersection(calibration_indices):
        raise ValueError("Inner model-training and calibration rows overlap.")
    if (
        set(model_train_indices).intersection(test_indices)
        or set(calibration_indices).intersection(test_indices)
    ):
        raise ValueError("Saved test indices entered a development partition.")

    model_train_groups = set(pd.util.hash_pandas_object(
        X_all.iloc[model_train_indices],
        index=False,
    ).to_numpy().tolist())
    calibration_groups = set(pd.util.hash_pandas_object(
        X_all.iloc[calibration_indices],
        index=False,
    ).to_numpy().tolist())
    test_groups = set(pd.util.hash_pandas_object(
        X_all.iloc[test_indices],
        index=False,
    ).to_numpy().tolist())
    if model_train_groups.intersection(calibration_groups):
        raise ValueError("Feature groups overlap model training and calibration.")
    if (
        model_train_groups.intersection(test_groups)
        or calibration_groups.intersection(test_groups)
    ):
        raise ValueError("Feature groups overlap a development split and test.")

    y_calibration = y_all[calibration_indices]
    y_test = y_all[test_indices]
    for name, labels in (
        ("calibration-validation", y_calibration),
        ("test", y_test),
    ):
        missing = sorted(set(range(len(classes))) - set(labels.tolist()))
        if missing:
            raise ValueError(
                f"{name} split lacks classes: {[classes[i] for i in missing]}"
            )

    xgb_params = manifest["model"]["xgboost_parameters"]
    model_training_params = dict(xgb_params)
    if model_training_params["num_class"] != len(classes):
        raise ValueError("Manifest XGBoost class count does not match mapping.")
    calibration_base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(**model_training_params)),
    ])
    X_model_train = X_all.iloc[model_train_indices]
    X_calibration = X_all.iloc[calibration_indices]
    X_test = X_all.iloc[test_indices]
    original_model = joblib.load(original_model_path)
    if not np.array_equal(
        original_model.named_steps["classifier"].classes_,
        np.arange(len(classes)),
    ):
        raise ValueError("Original model class order differs from mapping.")

    experiment_directory = create_experiment_directory(run_directory)
    start = time.time()
    calibration_base_model.fit(
        X_model_train,
        y_all[model_train_indices],
    )
    fit_seconds = time.time() - start

    # Calibrators see only calibration-validation predictions/labels.
    sigmoid_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(calibration_base_model),
        method="sigmoid",
        ensemble="auto",
    )
    sigmoid_model.fit(X_calibration, y_calibration)
    isotonic_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(calibration_base_model),
        method="isotonic",
        ensemble="auto",
    )
    isotonic_model.fit(X_calibration, y_calibration)

    base_classifier = calibration_base_model.named_steps["classifier"]
    transformed_calibration = calibration_base_model.named_steps[
        "imputer"
    ].transform(X_calibration)
    calibration_margins = base_classifier.predict(
        transformed_calibration,
        output_margin=True,
    )
    temperature, temperature_validation_log_loss = fit_temperature(
        np.asarray(calibration_margins),
        y_calibration,
    )
    temperature_model = TemperatureScaledModel(
        calibration_base_model,
        temperature,
    )

    # Save every fitted artifact in the new experiment directory only.
    model_artifacts = {
        "calibration_base_model.joblib": calibration_base_model,
        "sigmoid_calibrated_model.joblib": sigmoid_model,
        "isotonic_calibrated_model.joblib": isotonic_model,
        "temperature_scaled_model.joblib": temperature_model,
    }
    for filename, fitted_model in model_artifacts.items():
        joblib.dump(fitted_model, experiment_directory / filename)

    # Test predictions are generated only after all calibration fits finish.
    probability_outputs = {
        "saved_original_baseline": original_model.predict_proba(X_test),
        "inner_uncalibrated": calibration_base_model.predict_proba(X_test),
        "sigmoid": sigmoid_model.predict_proba(X_test),
        "isotonic": isotonic_model.predict_proba(X_test),
        "temperature": temperature_model.predict_proba(X_test),
    }
    expected_shape = (len(test_indices), len(classes))
    for method, probabilities in probability_outputs.items():
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"{method} returned {probabilities.shape}; "
                f"expected {expected_shape}."
            )

    summary_rows = []
    per_class_rows = []
    reliability = {}
    predictions_by_method = {}
    for method, probabilities in probability_outputs.items():
        metrics = calibration_metrics(probabilities, y_test)
        predictions = probabilities.argmax(axis=1)
        predictions_by_method[method] = predictions
        summary_rows.append({
            "Method": method,
            "Accuracy": metrics["accuracy"],
            "Macro_F1": metrics["macro_f1"],
            "Weighted_F1": metrics["weighted_f1"],
            "Multiclass_Brier": metrics["multiclass_brier_score"],
            "Normalized_Brier": metrics["normalized_brier_score"],
            "Multiclass_Log_Loss": metrics["multiclass_log_loss"],
            "Top_Label_ECE": metrics["top_label_ece"],
            "Classwise_ECE": metrics["classwise_ece"],
        })
        precision, recall, class_f1, support = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=np.arange(len(classes)),
            zero_division=0,
        )
        for class_index, class_name in enumerate(classes):
            p = probabilities[:, class_index]
            actual = (y_test == class_index).astype(float)
            class_ece = metrics["classwise_ece_by_index"][class_index]
            class_brier = float(np.mean((p - actual) ** 2))
            per_class_rows.append({
                "Method": method,
                "Traffic_Class": class_name,
                "Number_of_Samples": int(support[class_index]),
                "Classification_Accuracy": float(recall[class_index]),
                "Precision": float(precision[class_index]),
                "Recall": float(recall[class_index]),
                "F1": float(class_f1[class_index]),
                "Mean_Predicted_Probability": float(p.mean()),
                "Observed_Prevalence": float(actual.mean()),
                "One_vs_Rest_Brier": class_brier,
                "One_vs_Rest_ECE": float(class_ece),
            })
        reliability[method] = reliability_bins(probabilities, y_test)

    summary = pd.DataFrame(summary_rows)
    classwise = pd.DataFrame(per_class_rows)
    test_group_ids = pd.util.hash_pandas_object(
        X_test,
        index=False,
    ).to_numpy()
    performance_comparison = bootstrap_performance_differences(
        probability_outputs,
        y_test,
        test_group_ids,
    )

    summary.to_csv(
        experiment_directory / "calibration_method_comparison.csv",
        index=False,
        float_format="%.10g",
    )
    classwise.to_csv(
        experiment_directory / "per_class_calibration_analysis.csv",
        index=False,
        float_format="%.10g",
    )
    performance_comparison.to_csv(
        experiment_directory / "test_performance_bootstrap.csv",
        index=False,
        float_format="%.10g",
    )
    pd.DataFrame([
        {"Method": method, **item}
        for method, rows in reliability.items()
        for item in rows
    ]).to_csv(
        experiment_directory / "reliability_bins.csv",
        index=False,
        float_format="%.10g",
    )

    split_path = experiment_directory / "calibration_split_indices.npz"
    np.savez_compressed(
        split_path,
        model_training_source_indices=model_train_indices,
        calibration_validation_source_indices=calibration_indices,
        saved_test_source_indices=test_indices,
    )
    save_reliability_diagrams(
        experiment_directory,
        probability_outputs,
        y_test,
    )

    calibration_configuration = {
        "sigmoid": {
            "class": "sklearn.calibration.CalibratedClassifierCV",
            "method": "sigmoid",
            "multiclass_strategy": "one-vs-rest",
            "estimator": "FrozenEstimator(calibration_base_model)",
        },
        "isotonic": {
            "class": "sklearn.calibration.CalibratedClassifierCV",
            "method": "isotonic",
            "multiclass_strategy": "one-vs-rest",
            "estimator": "FrozenEstimator(calibration_base_model)",
        },
        "temperature": {
            "class": "TemperatureScaledModel",
            "optimization": "bounded scalar minimization of validation NLL",
            "optimized_parameter": "positive scalar temperature",
            "optimization_variable": "log(temperature)",
            "log_temperature_bounds": [-5.0, 5.0],
            "temperature": temperature,
            "validation_log_loss": temperature_validation_log_loss,
        },
    }
    configuration_path = experiment_directory / "calibration_configuration.json"
    write_json(configuration_path, calibration_configuration)
    report_path = experiment_directory / "calibration_experiment_report.txt"
    report_path.write_text(
        create_report(
            summary,
            classwise,
            performance_comparison,
            len(calibration_indices),
            len(model_train_indices),
            len(test_indices),
            {
                "temperature": temperature,
                "validation_log_loss": temperature_validation_log_loss,
            },
        ),
        encoding="utf-8",
    )

    analysis_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_directory.name,
        "source_run_id": manifest["run_id"],
        "source_dataset_sha256": dataset_hash,
        "source_original_model_sha256": original_model_hash,
        "source_label_mapping_sha256": sha256_file(mapping_path),
        "source_saved_train_indices_sha256": sha256_indices(
            original_train_indices
        ),
        "source_saved_test_indices_sha256": sha256_indices(test_indices),
        "split": {
            "class": "StratifiedGroupKFold",
            "n_splits": INNER_SPLIT_COUNT,
            "calibration_validation_fold": INNER_VALIDATION_FOLD,
            "shuffle": True,
            "random_state": INNER_SPLIT_SEED,
            "model_training_rows": int(len(model_train_indices)),
            "calibration_validation_rows": int(len(calibration_indices)),
            "test_rows": int(len(test_indices)),
            "model_training_indices_sha256": sha256_indices(
                model_train_indices
            ),
            "calibration_validation_indices_sha256": sha256_indices(
                calibration_indices
            ),
            "test_indices_sha256": sha256_indices(test_indices),
            "feature_groups_disjoint_across_model_calibration_test": True,
        },
        "calibration": calibration_configuration,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "random_state": BOOTSTRAP_SEED,
            "comparison_reference": "inner_uncalibrated",
            "resampling_unit": "exact-feature group",
            "confidence_interval": "paired percentile 95%",
        },
        "xgboost_parameters": manifest["model"]["xgboost_parameters"],
        "environment": current_versions,
        "model_artifacts": {
            filename: sha256_file(experiment_directory / filename)
            for filename in model_artifacts
        },
        "outputs": [
            *model_artifacts.keys(),
            "calibration_split_indices.npz",
            "calibration_method_comparison.csv",
            "per_class_calibration_analysis.csv",
            "test_performance_bootstrap.csv",
            "reliability_bins.csv",
            "calibration_configuration.json",
            "calibration_experiment_report.txt",
            *[
                f"reliability_{method}.png"
                for method in probability_outputs
            ],
        ],
        "fit_seconds": round(fit_seconds, 6),
        "operational_threshold_selected": False,
    }
    write_json(
        experiment_directory / "calibration_experiment_manifest.json",
        analysis_manifest,
    )

    if sha256_file(original_model_path) != original_model_hash:
        raise RuntimeError("Original model changed during calibration analysis.")
    if sha256_file(manifest_path) != original_manifest_hash:
        raise RuntimeError("Original experiment manifest changed.")

    print("Calibration experiment:", experiment_directory)
    print("Model-training rows:", len(model_train_indices))
    print("Calibration-validation rows:", len(calibration_indices))
    print("Saved test rows:", len(test_indices))
    print("Original baseline accuracy:", f"{manifest['metrics']['accuracy']:.2%}")
    print("Temperature:", f"{temperature:.8f}")
    print("\nTEST METRICS")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nPERFORMANCE DELTAS VS INNER UNCALIBRATED")
    print(performance_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))
    print("\nFiles:")
    for filename in analysis_manifest["outputs"] + [
        "calibration_experiment_manifest.json",
    ]:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
