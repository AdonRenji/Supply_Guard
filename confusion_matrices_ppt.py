import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import os

# ============================================================
# SUPPLYGUARD - CONFUSION MATRICES
# Random Forest and SVM
# ============================================================

os.makedirs("results", exist_ok=True)

labels = [
    "No Late-Delivery Risk",
    "Late-Delivery Risk"
]


# ============================================================
# RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = np.array([
    [13464, 2844],
    [2066, 17730]
])

print("Random Forest Confusion Matrix:")
print(rf_cm)

plt.figure(figsize=(7, 6))

rf_display = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=labels
)

rf_display.plot(
    values_format="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - Random Forest",
    fontsize=14
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    "results/random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# SVM CONFUSION MATRIX
# ============================================================

svm_cm = np.array([
    [14327, 1981],
    [8112, 11684]
])

print("\nSVM Confusion Matrix:")
print(svm_cm)

plt.figure(figsize=(7, 6))

svm_display = ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=labels
)

svm_display.plot(
    values_format="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - SVM",
    fontsize=14
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    "results/svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n============================================")
print("CONFUSION MATRICES GENERATED SUCCESSFULLY")
print("============================================")

print("\nFiles saved:")
print("1. results/random_forest_confusion_matrix.png")
print("2. results/svm_confusion_matrix.png")