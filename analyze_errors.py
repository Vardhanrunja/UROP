
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

folder = Path(
    r"C:\Users\veere\Downloads\dataset\Scenario B-ARFF"
)

# Load dataset
df = pd.read_csv(folder / "TimeBasedFeatures-Dataset-15s.csv")

X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

y = df.iloc[:, -1].astype(str)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Recreate the original train-test split
groups = pd.util.hash_pandas_object(
    X, index=False
).to_numpy()

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y_encoded, groups)
)

X_test = X.iloc[test_idx]
y_test = y_encoded[test_idx]

# Load saved XGBoost model
model = joblib.load(folder / "xgboost.joblib")

predictions = model.predict(X_test)

# Confusion matrix
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_))
)

matrix_df = pd.DataFrame(
    matrix,
    index=encoder.classes_,
    columns=encoder.classes_
)

print("\nCONFUSION MATRIX")
print(matrix_df.to_string())

# Check VPN-P2P errors
target = "VPN-P2P"
target_id = list(encoder.classes_).index(target)

actual_mask = y_test == target_id

print("\nVPN-P2P ANALYSIS")
print("Actual test samples:", actual_mask.sum())
print("Correct predictions:", (predictions[actual_mask] == target_id).sum())

wrong_predictions = predictions[actual_mask]

print("\nVPN-P2P misclassified as:")

for class_id, count in zip(
    *np.unique(wrong_predictions, return_counts=True)
):
    print(f"{encoder.classes_[class_id]}: {count}")

# Save confusion matrix
matrix_df.to_csv(folder / "confusion_matrix.csv")

print("\nSaved confusion_matrix.csv")
