import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_reproducibility import (
    find_latest_run,
    installed_versions,
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
CALIBRATION_BINS = 15

# These are fixed descriptive cutoffs for the requested examples and counts,
# not thresholds for abstention, routing, or operational decisions.
LOW_CONFIDENCE_DESCRIPTIVE_CUTOFF = 0.50
HIGH_CONFIDENCE_DESCRIPTIVE_CUTOFF = 0.90


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze confidence from a saved clean-XGBoost experiment."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Experiment run directory; defaults to the latest completed run.",
    )
    return parser.parse_args()


def create_analysis_directory(run_directory):
    root = Path(run_directory) / "confidence_analyses"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = root / f"confidence_analysis_{timestamp}"
    candidate = base
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = root / f"{base.name}_{suffix}"
            suffix += 1


def equal_width_bin_statistics(probabilities, outcomes, bin_count):
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    indices = np.minimum(
        np.digitize(probabilities, edges[1:-1], right=False),
        bin_count - 1,
    )
    rows = []
    for bin_index in range(bin_count):
        mask = indices == bin_index
        if not mask.any():
            rows.append({
                "bin_index": bin_index,
                "lower_bound": float(edges[bin_index]),
                "upper_bound": float(edges[bin_index + 1]),
                "count": 0,
                "mean_confidence": None,
                "accuracy": None,
            })
            continue
        rows.append({
            "bin_index": bin_index,
            "lower_bound": float(edges[bin_index]),
            "upper_bound": float(edges[bin_index + 1]),
            "count": int(mask.sum()),
            "mean_confidence": float(probabilities[mask].mean()),
            "accuracy": float(outcomes[mask].mean()),
        })
    return rows, indices


def top_label_ece(confidence, correct, bin_count):
    bins, _ = equal_width_bin_statistics(confidence, correct, bin_count)
    total = len(confidence)
    return float(sum(
        (item["count"] / total)
        * abs(item["accuracy"] - item["mean_confidence"])
        for item in bins
        if item["count"]
    )), bins


def classwise_ece(probabilities, y_true, bin_count):
    per_class = []
    for class_index in range(probabilities.shape[1]):
        outcomes = (y_true == class_index).astype(float)
        bins, _ = equal_width_bin_statistics(
            probabilities[:, class_index],
            outcomes,
            bin_count,
        )
        ece = float(sum(
            (item["count"] / len(y_true))
            * abs(item["accuracy"] - item["mean_confidence"])
            for item in bins
            if item["count"]
        ))
        per_class.append(ece)
    return float(np.mean(per_class)), per_class


def load_experiment(run_directory):
    run_directory = Path(run_directory).resolve()
    manifest_path = run_directory / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    dataset_hash = sha256_file(DATASET_PATH)
    if dataset_hash != manifest["dataset"]["sha256"]:
        raise ValueError(
            "Dataset SHA-256 does not match the saved experiment. "
            f"Expected {manifest['dataset']['sha256']}, got {dataset_hash}."
        )

    model_path = run_directory / manifest["model"]["artifact_file"]
    model_hash = sha256_file(model_path)
    if model_hash != manifest["model"]["sha256"]:
        raise ValueError(
            "Model SHA-256 does not match the saved experiment manifest."
        )

    mapping_path = run_directory / manifest["label_mapping_file"]
    mapping_hash = sha256_file(mapping_path)
    if mapping_hash != manifest["label_mapping_sha256"]:
        raise ValueError(
            "Label-mapping SHA-256 does not match the experiment manifest."
        )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    classes = mapping["classes"]
    class_to_index = mapping["class_to_index"]
    expected_mapping = {
        label: index for index, label in enumerate(classes)
    }
    if class_to_index != expected_mapping:
        raise ValueError("Saved class-to-index mapping is inconsistent.")

    split_path = run_directory / manifest["split"]["index_file"]
    with np.load(split_path, allow_pickle=False) as indices:
        test_indices = indices["test_source_indices"]
    if len(test_indices) != manifest["split"]["test_row_count"]:
        raise ValueError("Saved test-index count does not match the manifest.")
    if sha256_indices(test_indices) != (
        manifest["split"]["test_source_indices_sha256"]
    ):
        raise ValueError("Saved test indices do not match the manifest.")
    if (
        test_indices.ndim != 1
        or len(np.unique(test_indices)) != len(test_indices)
        or np.any(test_indices < 0)
    ):
        raise ValueError("Saved test indices are invalid.")

    df = pd.read_csv(DATASET_PATH)
    if np.any(test_indices >= len(df)):
        raise ValueError("Saved test indices exceed dataset row count.")
    feature_columns = manifest["preprocessing"]["feature_columns"]
    if df.columns[:-1].tolist() != feature_columns:
        raise ValueError(
            "Dataset feature columns or order differ from the saved model."
        )
    if df.columns[-1] != manifest["preprocessing"]["target_column"]:
        raise ValueError("Dataset target column differs from the manifest.")

    model = joblib.load(model_path)
    imputer = model.named_steps.get("imputer")
    classifier = model.named_steps.get("classifier")
    if imputer is None or classifier is None:
        raise ValueError("Saved model pipeline steps are not as expected.")
    fitted_features = getattr(imputer, "feature_names_in_", None)
    if fitted_features is None or list(fitted_features) != feature_columns:
        raise ValueError(
            "Saved model feature columns do not match the experiment."
        )
    expected_parameters = manifest["model"]["xgboost_parameters"]
    actual_parameters = classifier.get_params()
    mismatches = {
        key: (value, actual_parameters.get(key))
        for key, value in expected_parameters.items()
        if actual_parameters.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Saved model parameters do not match the manifest: {mismatches}"
        )
    if not np.array_equal(
        classifier.classes_,
        np.arange(len(classes)),
    ):
        raise ValueError("Model class IDs do not match saved label mapping.")

    current_versions = installed_versions()
    expected_versions = manifest["environment"]
    version_mismatches = {
        key: (expected, current_versions.get(key))
        for key, expected in expected_versions.items()
        if current_versions.get(key) != expected
    }
    if version_mismatches:
        raise RuntimeError(
            "Current environment differs from the saved experiment: "
            f"{version_mismatches}"
        )

    return {
        "run_directory": run_directory,
        "manifest": manifest,
        "mapping": mapping,
        "dataset_hash": dataset_hash,
        "model_hash": model_hash,
        "mapping_hash": mapping_hash,
        "model": model,
        "dataframe": df,
        "test_indices": test_indices,
        "feature_columns": feature_columns,
    }


def write_confidence_plot(path, confidence, correct, classes, y_true):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0.0, 1.0, 21)
    axes[0].hist(
        confidence[correct],
        bins=bins,
        alpha=0.72,
        label=f"Correct (n={int(correct.sum())})",
        color="#2a9d8f",
    )
    axes[0].hist(
        confidence[~correct],
        bins=bins,
        alpha=0.72,
        label=f"Incorrect (n={int((~correct).sum())})",
        color="#e76f51",
    )
    axes[0].set(
        title="Maximum confidence distribution",
        xlabel="Maximum class probability",
        ylabel="Record count",
        xlim=(0.0, 1.0),
    )
    axes[0].legend()

    class_confidence = [
        confidence[y_true == class_index]
        for class_index in range(len(classes))
    ]
    axes[1].boxplot(
        class_confidence,
        tick_labels=classes,
        showfliers=False,
    )
    axes[1].tick_params(axis="x", labelrotation=65, labelsize=8)
    axes[1].set(
        title="Confidence by actual traffic class",
        ylabel="Maximum class probability",
        ylim=(0.0, 1.02),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_top_label_reliability_plot(path, bins, ece):
    populated = [item for item in bins if item["count"]]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        [item["mean_confidence"] for item in populated],
        [item["accuracy"] for item in populated],
        marker="o",
        color="#264653",
        label="Model",
    )
    for item in populated:
        ax.annotate(
            str(item["count"]),
            (item["mean_confidence"], item["accuracy"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set(
        title=f"Top-label reliability (ECE={ece:.4f})",
        xlabel="Mean maximum confidence in bin",
        ylabel="Accuracy in bin",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_classwise_reliability_plot(path, probabilities, y_true, classes):
    bin_count = CALIBRATION_BINS
    columns = 4
    rows = int(np.ceil(len(classes) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 3.8),
        squeeze=False,
    )
    for class_index, class_name in enumerate(classes):
        ax = axes.flat[class_index]
        outcomes = (y_true == class_index).astype(float)
        bins, _ = equal_width_bin_statistics(
            probabilities[:, class_index],
            outcomes,
            bin_count,
        )
        populated = [item for item in bins if item["count"]]
        ece = float(sum(
            (item["count"] / len(y_true))
            * abs(item["accuracy"] - item["mean_confidence"])
            for item in populated
        ))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.plot(
            [item["mean_confidence"] for item in populated],
            [item["accuracy"] for item in populated],
            marker="o",
            color="#2a9d8f",
        )
        ax.set(
            title=f"{class_name} (OvR ECE={ece:.3f})",
            xlabel="Mean predicted class probability",
            ylabel="Observed class frequency",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
        )
    for ax in axes.flat[len(classes):]:
        ax.set_visible(False)
    fig.suptitle("One-vs-rest classwise reliability", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_report(
    run_id,
    classes,
    source_indices,
    y_true,
    predicted,
    confidence,
    correct,
    class_analysis,
    brier,
    normalized_brier,
    logloss,
    top_ece,
    classwise_ece,
    bins,
):
    weakest = class_analysis.sort_values(
        ["Classification_Accuracy", "Number_of_Samples"],
        ascending=[True, False],
    ).head(4)
    low_conf_correct = np.flatnonzero(
        correct & (confidence < LOW_CONFIDENCE_DESCRIPTIVE_CUTOFF)
    )
    low_conf_correct = low_conf_correct[
        np.argsort(confidence[low_conf_correct])
    ][:20]
    high_conf_incorrect = np.flatnonzero(
        ~correct & (confidence >= HIGH_CONFIDENCE_DESCRIPTIVE_CUTOFF)
    )
    high_conf_incorrect = high_conf_incorrect[
        np.argsort(-confidence[high_conf_incorrect])
    ][:20]

    lines = [
        f"Confidence analysis for {run_id}",
        "=" * (23 + len(run_id)),
        "",
        "Scope and integrity",
        f"- Test records: {len(y_true)} across {len(classes)} classes.",
        "- Loaded the saved model, manifest, label mapping, and test indices.",
        "- Dataset and model SHA-256 hashes were verified before inference.",
        "- No training or model modification was performed.",
        "",
        "Overall findings",
        f"- Accuracy: {correct.mean() * 100:.2f}% "
        f"({int(correct.sum())}/{len(correct)}).",
        f"- Mean maximum confidence: {confidence.mean():.4f}; "
        f"median: {np.median(confidence):.4f}.",
        f"- Confidence range: {confidence.min():.4f} to "
        f"{confidence.max():.4f}.",
        f"- Multiclass Brier score (sum over classes): {brier:.6f}.",
        f"- Brier score normalized by 2: {normalized_brier:.6f}.",
        f"- Multiclass log loss: {logloss:.6f}.",
        f"- Top-label ECE ({CALIBRATION_BINS} equal-width bins): "
        f"{top_ece:.6f}.",
        f"- Macro classwise one-vs-rest ECE: {classwise_ece:.6f}.",
        "",
        "Confidence interpretation",
        "- Maximum class probability is model confidence, not correctness "
        "or a guarantee of correctness.",
        "- The 0.50 low-confidence and 0.90 high-confidence cutoffs below "
        "are fixed descriptive reporting conventions only.",
        "- They were not optimized on this test set and must not be used as "
        "operational thresholds.",
        "",
        "Weak traffic categories (lowest observed accuracy; descriptive)",
    ]
    for row in weakest.itertuples(index=False):
        lines.append(
            f"- {row.Traffic_Class}: accuracy "
            f"{row.Classification_Accuracy:.2%}, n={row.Number_of_Samples}, "
            f"mean confidence {row.Mean_Confidence:.4f}."
        )

    high_count = int((~correct & (
        confidence >= HIGH_CONFIDENCE_DESCRIPTIVE_CUTOFF
    )).sum())
    lines.extend([
        "",
        f"High-confidence incorrect predictions "
        f"(descriptive confidence >= "
        f"{HIGH_CONFIDENCE_DESCRIPTIVE_CUTOFF:.2f}): {high_count}",
    ])
    if not len(high_conf_incorrect):
        lines.append("- None found.")
    for index in high_conf_incorrect:
        lines.append(
            f"- Source row {int(source_indices[index])}: "
            f"actual={classes[y_true[index]]}, "
            f"predicted={classes[predicted[index]]}, "
            f"confidence={confidence[index]:.4f}."
        )

    low_count = int((correct & (
        confidence < LOW_CONFIDENCE_DESCRIPTIVE_CUTOFF
    )).sum())
    lines.extend([
        "",
        f"Low-confidence correct predictions "
        f"(descriptive confidence < "
        f"{LOW_CONFIDENCE_DESCRIPTIVE_CUTOFF:.2f}): {low_count}",
    ])
    if not len(low_conf_correct):
        lines.append("- None found.")
    for index in low_conf_correct:
        lines.append(
            f"- Source row {int(source_indices[index])}: "
            f"class={classes[y_true[index]]}, "
            f"confidence={confidence[index]:.4f}."
        )

    populated = [item for item in bins if item["count"]]
    lines.extend([
        "",
        "Calibration-bin observations",
    ])
    for item in populated:
        lines.append(
            f"- [{item['lower_bound']:.3f}, {item['upper_bound']:.3f}"
            f"{']' if item['upper_bound'] == 1.0 else ')'}: "
            f"n={item['count']}, mean confidence="
            f"{item['mean_confidence']:.4f}, "
            f"accuracy={item['accuracy']:.4f}."
        )

    lines.extend([
        "",
        "Recommendations",
        "- Treat confidence as a review signal, not truth. The observed "
        "high-confidence mistakes show that confident errors occur.",
        "- For an operational abstention or escalation policy, define a "
        "separate calibration/validation set from training data. Choose and "
        "freeze any threshold there using predeclared costs and coverage "
        "targets; evaluate the frozen policy once on an untouched test set.",
        "- If probabilities need calibration, fit the calibrator only on "
        "validation data (or out-of-fold training predictions), then assess "
        "calibration on the untouched test set.",
        "- Check calibration and error rates by traffic class and VPN status; "
        "aggregate confidence can obscure weak categories.",
        "- Do not alter DSCP mappings or infer network QoS benefit from this "
        "offline confidence analysis.",
        "",
        "Calibration definitions",
        "- Multiclass Brier: mean over records of the sum across classes of "
        "(probability - one-hot target)^2; normalized score divides by 2.",
        "- Top-label ECE: sample-weighted absolute gap between correctness "
        "and maximum confidence in 15 equal-width probability bins.",
        "- Classwise ECE: macro-average over 14 one-vs-rest ECE values; each "
        "class uses the same 15 bins and sample-weighted bin gaps.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    run_directory = (
        args.run_dir.resolve()
        if args.run_dir
        else find_latest_run(RUNS_ROOT)
    )
    experiment = load_experiment(run_directory)
    manifest = experiment["manifest"]
    mapping = experiment["mapping"]
    classes = mapping["classes"]
    class_to_index = mapping["class_to_index"]
    df = experiment["dataframe"]
    test_indices = experiment["test_indices"]
    feature_columns = experiment["feature_columns"]

    X_test = (
        df.iloc[test_indices]
        .loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    actual_labels = df.iloc[test_indices, -1].astype(str).to_numpy()
    unknown_labels = sorted(set(actual_labels) - set(class_to_index))
    if unknown_labels:
        raise ValueError(f"Test data contains unmapped labels: {unknown_labels}")
    y_true = np.asarray(
        [class_to_index[label] for label in actual_labels],
        dtype=np.int64,
    )

    probabilities = experiment["model"].predict_proba(X_test)
    expected_shape = (len(test_indices), len(classes))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected probability matrix {expected_shape}, "
            f"got {probabilities.shape}."
        )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("Model returned invalid class probabilities.")

    predicted_indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted_indices == y_true
    brier = float(np.mean(np.sum(
        (probabilities - np.eye(len(classes))[y_true]) ** 2,
        axis=1,
    )))
    normalized_brier = brier / 2.0
    logloss = float(-np.log(
        np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    ).mean())
    top_ece, reliability_bins = top_label_ece(
        confidence,
        correct.astype(float),
        CALIBRATION_BINS,
    )
    macro_classwise_ece, per_class_ece = classwise_ece(
        probabilities,
        y_true,
        CALIBRATION_BINS,
    )
    predicted_labels = np.asarray(classes)[predicted_indices]

    analysis_directory = create_analysis_directory(run_directory)
    probability_columns = {
        f"Probability_{label}": probabilities[:, class_index]
        for class_index, label in enumerate(classes)
    }
    predictions = pd.DataFrame({
        "Original_Source_Row_Index": test_indices.astype(np.int64),
        "Actual_Traffic_Class": actual_labels,
        "Predicted_Traffic_Class": predicted_labels,
        "Prediction_Confidence": confidence,
        "Prediction_Status": np.where(correct, "correct", "incorrect"),
        **probability_columns,
    })
    predictions_path = analysis_directory / "confidence_predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.10g")

    class_rows = []
    for class_index, class_name in enumerate(classes):
        mask = y_true == class_index
        if not mask.any():
            raise ValueError(f"No test examples for class {class_name}.")
        errors = mask & ~correct
        class_rows.append({
            "Traffic_Class": class_name,
            "Number_of_Samples": int(mask.sum()),
            "Classification_Accuracy": float(correct[mask].mean()),
            "Mean_Confidence": float(confidence[mask].mean()),
            "Median_Confidence": float(np.median(confidence[mask])),
            "Incorrect_High_Confidence_Predictions": int((
                errors
                & (confidence >= HIGH_CONFIDENCE_DESCRIPTIVE_CUTOFF)
            ).sum()),
        })
    class_analysis = pd.DataFrame(class_rows)
    class_analysis_path = analysis_directory / "class_confidence_analysis.csv"
    class_analysis.to_csv(
        class_analysis_path,
        index=False,
        float_format="%.10g",
    )

    classwise_details = []
    for index, class_name in enumerate(classes):
        classwise_details.append({
            "class": class_name,
            "one_vs_rest_ece": per_class_ece[index],
        })
    calibration = {
        "multiclass_brier_score": brier,
        "normalized_brier_score": normalized_brier,
        "multiclass_log_loss": logloss,
        "top_label_ece": top_ece,
        "classwise_ece": macro_classwise_ece,
        "classwise_ece_by_class": classwise_details,
        "number_of_calibration_bins": CALIBRATION_BINS,
        "binning": "15 equal-width bins over [0, 1], final bin includes 1",
        "top_label_ece_definition": (
            "Sample-weighted absolute difference between mean maximum "
            "confidence and accuracy within each bin."
        ),
        "classwise_ece_definition": (
            "Macro-average of 14 one-vs-rest ECE values; each class uses "
            "sample-weighted absolute probability/frequency gaps across "
            "the same equal-width bins."
        ),
        "brier_score_definition": (
            "Mean over records of the sum across classes of squared "
            "probability minus one-hot target; normalized score divides "
            "this value by 2."
        ),
        "dataset_sha256": experiment["dataset_hash"],
        "model_sha256": experiment["model_hash"],
        "run_id": manifest["run_id"],
        "test_record_count": int(len(test_indices)),
        "descriptive_only_cutoffs": {
            "low_confidence_correct_below": (
                LOW_CONFIDENCE_DESCRIPTIVE_CUTOFF
            ),
            "high_confidence_incorrect_at_or_above": (
                HIGH_CONFIDENCE_DESCRIPTIVE_CUTOFF
            ),
            "operational_threshold_selected": False,
        },
    }
    calibration_path = analysis_directory / "calibration_analysis.json"
    write_json(calibration_path, calibration)

    report = build_report(
        manifest["run_id"],
        classes,
        test_indices,
        y_true,
        predicted_indices,
        confidence,
        correct,
        class_analysis,
        brier,
        normalized_brier,
        logloss,
        top_ece,
        macro_classwise_ece,
        reliability_bins,
    )
    report_path = analysis_directory / "confidence_analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")

    confidence_plot_path = analysis_directory / "confidence_distribution.png"
    write_confidence_plot(
        confidence_plot_path,
        confidence,
        correct,
        classes,
        y_true,
    )
    top_reliability_path = (
        analysis_directory / "reliability_diagram_top_label.png"
    )
    write_top_label_reliability_plot(
        top_reliability_path,
        reliability_bins,
        top_ece,
    )
    class_reliability_path = (
        analysis_directory / "reliability_diagram_classwise.png"
    )
    write_classwise_reliability_plot(
        class_reliability_path,
        probabilities,
        y_true,
        classes,
    )

    analysis_manifest = {
        "schema_version": 1,
        "analysis_id": analysis_directory.name,
        "source_run_id": manifest["run_id"],
        "source_dataset_sha256": experiment["dataset_hash"],
        "source_model_sha256": experiment["model_hash"],
        "source_label_mapping_sha256": experiment["mapping_hash"],
        "source_test_indices_sha256": manifest["split"][
            "test_source_indices_sha256"
        ],
        "test_record_count": int(len(test_indices)),
        "class_count": int(len(classes)),
        "metrics_file": calibration_path.name,
        "outputs": [
            predictions_path.name,
            class_analysis_path.name,
            calibration_path.name,
            report_path.name,
            confidence_plot_path.name,
            top_reliability_path.name,
            class_reliability_path.name,
        ],
        "environment": installed_versions(),
    }
    analysis_manifest_path = analysis_directory / "analysis_manifest.json"
    write_json(analysis_manifest_path, analysis_manifest)

    print("Source experiment:", manifest["run_id"])
    print("Confidence analysis directory:", analysis_directory)
    print("Test rows:", len(test_indices))
    print("Accuracy:", f"{correct.mean() * 100:.2f}%")
    print("Correct predictions:", int(correct.sum()))
    print("Incorrect predictions:", int((~correct).sum()))
    print("Multiclass Brier:", f"{brier:.6f}")
    print("Normalized Brier:", f"{normalized_brier:.6f}")
    print("Multiclass log loss:", f"{logloss:.6f}")
    print("Top-label ECE:", f"{top_ece:.6f}")
    print("Classwise ECE:", f"{macro_classwise_ece:.6f}")
    print("Files:")
    for filename in analysis_manifest["outputs"] + [
        analysis_manifest_path.name,
    ]:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
