import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


PACKAGE_VERSIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "xgboost": "xgboost",
    "joblib": "joblib",
    "scipy": "scipy",
    "threadpoolctl": "threadpoolctl",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_indices(indices):
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def installed_versions():
    versions = {"python": platform.python_version()}
    for key, distribution in PACKAGE_VERSIONS.items():
        try:
            versions[key] = version(distribution)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Required package is not installed: {distribution}"
            ) from error
    return versions


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def load_clean_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    raw_df = pd.read_csv(dataset_path)
    if raw_df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature and a label.")

    feature_columns = raw_df.columns[:-1].tolist()
    target_column = raw_df.columns[-1]
    X = raw_df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = raw_df.iloc[:, -1].astype(str)

    signatures = pd.util.hash_pandas_object(X, index=False)
    group_data = pd.DataFrame({"signature": signatures, "label": y})
    label_counts = group_data.groupby("signature")["label"].nunique()
    conflicting_groups = label_counts[label_counts > 1].index
    keep = ~signatures.isin(conflicting_groups)

    source_indices = np.flatnonzero(keep.to_numpy()).astype(np.int64)
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    if X.empty:
        raise ValueError("No rows remain after removing conflicting labels.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    train_indices, test_indices = next(
        splitter.split(X, y_encoded, groups)
    )

    return {
        "raw_df": raw_df,
        "X": X,
        "y": y,
        "encoder": encoder,
        "y_encoded": y_encoded,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_source_indices": source_indices[train_indices],
        "test_source_indices": source_indices[test_indices],
        "clean_source_indices": source_indices,
        "conflicting_group_count": int(len(conflicting_groups)),
        "removed_row_count": int((~keep).sum()),
        "feature_columns": feature_columns,
        "target_column": target_column,
    }


def create_run_directory(runs_root):
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"clean_xgboost_{timestamp}"
    candidate = runs_root / base_name
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = runs_root / f"{base_name}_{suffix}"
            suffix += 1


def find_latest_run(runs_root):
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(
            f"No reproducible runs found at {runs_root}. "
            "Run train_xgboost_clean.py first."
        )
    candidates = sorted(
        path for path in runs_root.glob("clean_xgboost_*")
        if path.is_dir() and (path / "experiment_manifest.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed clean-XGBoost run exists in {runs_root}. "
            "Run train_xgboost_clean.py first."
        )
    return candidates[-1]
