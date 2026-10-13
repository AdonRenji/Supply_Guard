
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

file_path = (
    "results/feature_group_experiments/"
    "chronological_test_predictions.csv"
)

df = pd.read_csv(file_path)

y_true = df["Actual Risk"]
probabilities = df["Predicted Probability"]

print("\nSUPPLYGUARD - THRESHOLD COMPARISON")
print("-" * 90)
print(
    f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} "
    f"{'Recall':>10} {'F1':>10} {'FP':>7} {'FN':>7}"
)

for threshold in [0.30, 0.35, 0.41, 0.45, 0.50, 0.55]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:10.2f} "
        f"{accuracy_score(y_true, predictions):10.4f} "
        f"{precision_score(y_true, predictions, zero_division=0):10.4f} "
        f"{recall_score(y_true, predictions, zero_division=0):10.4f} "
        f"{f1_score(y_true, predictions, zero_division=0):10.4f} "
        f"{fp:7d} {fn:7d}"
    )

print("\nComparison complete. No model files were changed.")
