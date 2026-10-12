import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SUPPLYGUARD - RANDOM FOREST VS SVM COMPARISON
# ============================================================

print("=" * 70)
print("SUPPLYGUARD - MODEL COMPARISON")
print("=" * 70)


# ============================================================
# 1. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = "results"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 2. MODEL RESULTS
# ============================================================

# Random Forest:
# Final 31-feature model
# 500 trees
# Full training set: 144,415 records
# Test set: 36,104 records

random_forest = {

    "Model": "Random Forest",

    "Accuracy": 86.40,

    "Precision": 86.18,

    "Recall": 89.56,

    "F1 Score": 87.84,

    "ROC-AUC": 93.13,

    "Training Time (seconds)": 228.01,

    "Training Samples": 144415,

    "Testing Samples": 36104

}


# SVM:
# RBF kernel
# C = 1.0
# Gamma = scale
# Trained using a stratified 30,000-record subset
# Test set: 36,104 records

svm = {

    "Model": "SVM",

    "Accuracy": 72.42,

    "Precision": 78.47,

    "Recall": 68.50,

    "F1 Score": 73.15,

    "ROC-AUC": 79.21,

    "Training Time (seconds)": 920.35,

    "Training Samples": 30000,

    "Testing Samples": 36104

}


# ============================================================
# 3. CREATE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    [
        random_forest,
        svm
    ]
)


# ============================================================
# 4. DISPLAY COMPARISON
# ============================================================

print("\nModel Comparison:")
print("-" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 5. SAVE COMPARISON CSV
# ============================================================

comparison_csv = os.path.join(
    OUTPUT_FOLDER,
    "model_comparison.csv"
)


comparison_df.to_csv(
    comparison_csv,
    index=False
)


print(
    f"\nComparison table saved to: "
    f"{comparison_csv}"
)


# ============================================================
# 6. METRIC COMPARISON
# ============================================================

metrics = [

    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"

]


metric_data = comparison_df[
    ["Model"] + metrics
]


# ============================================================
# 7. CREATE BAR CHART
# ============================================================

print(
    "\nCreating model comparison chart..."
)


ax = metric_data.set_index(
    "Model"
)[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)


ax.set_title(
    "SupplyGuard - Random Forest vs SVM"
)


ax.set_xlabel(
    "Model"
)


ax.set_ylabel(
    "Performance (%)"
)


ax.set_ylim(
    0,
    100
)


ax.legend(
    title="Metric"
)


plt.xticks(
    rotation=0
)


plt.tight_layout()


chart_path = os.path.join(
    OUTPUT_FOLDER,
    "model_comparison.png"
)


plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Comparison chart saved to: "
    f"{chart_path}"
)


# ============================================================
# 8. TRAINING TIME COMPARISON
# ============================================================

print(
    "\nCreating training time comparison..."
)


time_df = comparison_df[
    [
        "Model",
        "Training Time (seconds)"
    ]
]


ax = time_df.set_index(
    "Model"
).plot(
    kind="bar",
    figsize=(8, 5),
    legend=False
)


ax.set_title(
    "SupplyGuard - Model Training Time"
)


ax.set_xlabel(
    "Model"
)


ax.set_ylabel(
    "Training Time (seconds)"
)


plt.xticks(
    rotation=0
)


plt.tight_layout()


time_chart_path = os.path.join(
    OUTPUT_FOLDER,
    "model_training_time_comparison.png"
)


plt.savefig(
    time_chart_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Training time chart saved to: "
    f"{time_chart_path}"
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)

print("\nRandom Forest:")
print("Accuracy:  86.40%")
print("Precision: 86.18%")
print("Recall:    89.56%")
print("F1 Score:  87.84%")
print("ROC-AUC:   93.13%")

print("\nSVM:")
print("Accuracy:  72.42%")
print("Precision: 78.47%")
print("Recall:    68.50%")
print("F1 Score:  73.15%")
print("ROC-AUC:   79.21%")

print("\nGenerated files:")

print(
    "1. results/model_comparison.csv"
)

print(
    "2. results/model_comparison.png"
)

print(
    "3. results/model_training_time_comparison.png"
)

print("\nImportant:")
print(
    "SVM was trained using a stratified "
    "30,000-record subset because of computational cost."
)

print(
    "Random Forest was trained using the "
    "full 144,415-record training set."
)

print("=" * 70)