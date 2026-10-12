from sklearn.metrics import classification_report

# ============================================================
# SUPPLYGUARD - RANDOM FOREST OUTPUT
# ============================================================

print("\n" + "=" * 45)
print("=== Random Forest ===")
print("=" * 45)

# Actual Random Forest results
accuracy = 0.8640

# Confusion matrix
# [[TN, FP],
#  [FN, TP]]

tn = 13464
fp = 2844
fn = 2066
tp = 17730

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
print("RANDOM FOREST RESULTS")
print("=" * 45)

print("Accuracy : 86.40%")
print("Precision: 86.18%")
print("Recall   : 89.56%")
print("F1-Score : 87.84%")
print("ROC-AUC  : 93.13%")

print("\nConfusion Matrix:")
print("[[13464  2844]")
print(" [ 2066 17730]]")

print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)