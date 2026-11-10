
import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

# 1. Load dataset
folder = Path(r"C:\Users\veere\Downloads\dataset\Scenario B-ARFF")
df = pd.read_csv(folder / "TimeBasedFeatures-Dataset-15s.csv")

X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
y = df.iloc[:, -1].astype(str)

# 2. Remove conflicting feature groups
signatures = pd.util.hash_pandas_object(X, index=False)

groups_data = pd.DataFrame({
    "signature": signatures,
    "label": y
})

label_counts = groups_data.groupby("signature")["label"].nunique()
conflicts = label_counts[label_counts > 1].index

keep = ~signatures.isin(conflicts)

X = X.loc[keep].reset_index(drop=True)
y = y.loc[keep].reset_index(drop=True)

print("Clean dataset records:", len(X))

# 3. Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 4. Group-based train-test split
groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Define models
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

results = []

# 6. Train and evaluate
for name, classifier in models.items():

    print("\nTraining:", name)
    start = time.time()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    elapsed = time.time() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro-F1:", round(macro_f1, 4))
    print("Weighted-F1:", round(weighted_f1, 4))

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "Macro-F1": round(macro_f1, 4),
        "Weighted-F1": round(weighted_f1, 4),
        "Time": round(elapsed, 2)
    })

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )

    with open(
        folder / (name.lower().replace(" ", "_") + "_clean_report.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    joblib.dump(
        pipeline,
        folder / (name.lower().replace(" ", "_") + "_clean.joblib")
    )

# 7. Compare results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Macro-F1", ascending=False)

print("\nFINAL CLEAN MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(folder / "clean_model_comparison.csv", index=False)

print("\nAll models completed successfully!")
print("Results saved in clean_model_comparison.csv")
