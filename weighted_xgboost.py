
import pandas as pd
import numpy as np
import time

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

# Load dataset
folder = Path(r"C:\Users\veere\Downloads\dataset\Scenario B-ARFF")
df = pd.read_csv(folder / "TimeBasedFeatures-Dataset-15s.csv")

X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
y = df.iloc[:, -1].astype(str)

# Remove conflicting feature groups
signatures = pd.util.hash_pandas_object(X, index=False)
labels = pd.DataFrame({"group": signatures, "label": y})

conflicts = labels.groupby("group")["label"].nunique()
conflicts = conflicts[conflicts > 1].index

keep = ~signatures.isin(conflicts)
X = X.loc[keep].reset_index(drop=True)
y = y.loc[keep].reset_index(drop=True)

# Encode classes
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Same grouped train-test split
groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

# Common XGBoost settings
def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=len(encoder.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        ))
    ])

# Train normal XGBoost
print("\nTraining normal XGBoost...")
normal = create_model()
normal.fit(X_train, y_train)
normal_predictions = normal.predict(X_test)

# Calculate class weights using training data only
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (
    len(class_counts) * class_counts
)
sample_weights = class_weights[y_train]

# Train class-weighted XGBoost
print("Training class-weighted XGBoost...")
weighted = create_model()

start = time.time()
weighted.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)
elapsed = time.time() - start

weighted_predictions = weighted.predict(X_test)

# Compare results
print("\n========== MODEL COMPARISON ==========")

for name, predictions in [
    ("Normal XGBoost", normal_predictions),
    ("Weighted XGBoost", weighted_predictions)
]:
    print("\n", name)
    print("Accuracy:",
          round(accuracy_score(y_test, predictions) * 100, 2))
    print("Macro-F1:",
          round(f1_score(y_test, predictions, average="macro"), 4))

# Detailed weighted report
report = classification_report(
    y_test,
    weighted_predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
)

print("\nWEIGHTED XGBOOST CLASSIFICATION REPORT")
print(report)

# Save outputs
weighted_report = folder / "weighted_xgboost_report.txt"
weighted_report.write_text(report, encoding="utf-8")

normal_report = classification_report(
    y_test,
    normal_predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
)

(foler_report if False else normal_report)

normal_path = folder / "normal_xgboost_report.txt"
normal_path.write_text(normal_report, encoding="utf-8")

import joblib
joblib.dump(weighted, folder / "weighted_xgboost.joblib")

print("\nWeighted model saved successfully!")
print("Training time:", round(elapsed, 2), "seconds")
