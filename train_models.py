
import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from xgboost import XGBClassifier

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = Path(
    r"C:\Users\veere\Downloads\dataset\Scenario B-ARFF\TimeBasedFeatures-Dataset-15s.csv"
)

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# Last column is the traffic class
X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
y = df.iloc[:, -1].astype(str)

# Handle infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing labels
valid = y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

print("Features:", X.shape[1])
print("Traffic classes:", y.nunique())
print("Records:", len(X))

# ==========================================
# 2. ENCODE CLASS LABELS
# ==========================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nTraffic classes:")
for i, name in enumerate(encoder.classes_):
    print(i, name)

# Group identical feature combinations together.
# This prevents matching feature vectors from
# appearing in both training and testing sets.

groups = pd.util.hash_pandas_object(
    X, index=False
).to_numpy()

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

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

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
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
    )
}

# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

results = []
output_folder = file_path.parent
report_file = output_folder / "classification_reports.txt"

with open(report_file, "w", encoding="utf-8") as report:

    report.write("MODEL CLASSIFICATION REPORTS\n")
    report.write("=" * 60 + "\n")

    for name, model in models.items():

        print("\n" + "=" * 50)
        print("Training:", name)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", model)
        ])

        start = time.time()

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        elapsed = time.time() - start

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro"
        )
        weighted_f1 = f1_score(
            y_test, predictions, average="weighted"
        )

        print("Accuracy:", round(accuracy * 100, 2), "%")
        print("Macro F1:", round(macro_f1, 4))
        print("Weighted F1:", round(weighted_f1, 4))
        print("Training time:", round(elapsed, 2), "seconds")

        results.append({
            "Model": name,
            "Accuracy": round(accuracy * 100, 2),
            "Macro F1": round(macro_f1, 4),
            "Weighted F1": round(weighted_f1, 4),
            "Time (seconds)": round(elapsed, 2)
        })

        # Save detailed classification report
        class_report = classification_report(
            y_test,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )

        report.write("\n\n" + name + "\n")
        report.write(class_report)

        # Save trained model
        model_path = output_folder / (
            name.lower().replace(" ", "_") + ".joblib"
        )

        joblib.dump(pipeline, model_path)

        print("Model saved:", model_path.name)

# ==========================================
# 6. COMPARE ALL MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n\nFINAL MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

# Save comparison
results_df.to_csv(
    output_folder / "model_comparison.csv",
    index=False
)

print("\nResults saved in:", output_folder)
print("Comparison: model_comparison.csv")
print("Detailed reports: classification_reports.txt")
print("\nTRAINING COMPLETED SUCCESSFULLY!")
