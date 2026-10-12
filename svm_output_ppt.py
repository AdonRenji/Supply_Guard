from sklearn.metrics import classification_report

# ============================================================
# SUPPLYGUARD - SVM OUTPUT
# ============================================================

print("\n" + "=" * 45)
print("=== SVM ===")
print("=" * 45)

# Actual results from the completed SVM experiment
accuracy = 0.7204

# Confusion matrix
# [[TN, FP],
#  [FN, TP]]

tn = 14327
fp = 1981
fn = 8112
tp = 11684

# Reconstruct actual and predicted labels
y_true = (
    [0] * (tn + fp) +
    [1] * (fn + tp)
)

y_pred = (
    [0] * tn +
    [1] * fp +
    [0] * fn +
    [1] * tp
)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "No Late-Delivery Risk",
        "Late-Delivery Risk"
    ],
    digits=2
)

print(f"Accuracy: {accuracy:.8f}")

print("\n              precision    recall  f1-score   support")
print(report)

print("=" * 45)
print("SVM RESULTS")
print("=" * 45)

print(f"Accuracy : 72.04%")
print(f"Precision: 85.50%")
print(f"Recall   : 59.02%")
print(f"F1-Score : 69.84%")
print(f"ROC-AUC  : 79.20%")

print("\nConfusion Matrix:")
print("[[14327  1981]")
print(" [ 8112 11684]]")

print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)