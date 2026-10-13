
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

path = (
    "results/feature_group_experiments/"
    "chronological_test_predictions.csv"
)

df = pd.read_csv(path)
y_true = df["Actual Risk"]
probabilities = df["Predicted Probability"]

print("\nSUPPLYGUARD - LOWER THRESHOLD EXPERIMENT")
print("-" * 85)
print(
    f"{'Threshold':>10} {'Recall':>9} {'Precision':>10} "
    f"{'Accuracy':>10} {'F1':>9} {'Missed':>9} {'False alarms':>13}"
)

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.41]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:10.2f} "
        f"{recall_score(y_true, predictions):9.4f} "
        f"{precision_score(y_true, predictions, zero_division=0):10.4f} "
        f"{accuracy_score(y_true, predictions):10.4f} "
        f"{f1_score(y_true, predictions):9.4f} "
        f"{fn:9d} {fp:13d}"
    )

print("\nEvaluation only: no model files were changed.")
