import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# SUPPLYGUARD - RANDOM FOREST vs SVM
# TEST SET PERFORMANCE
# ============================================================

models = ["Random Forest", "SVM"]

accuracy = [86.40, 72.04]
precision = [86.18, 85.50]
recall = [89.56, 59.02]
f1_score = [87.84, 69.84]
roc_auc = [93.13, 79.20]


# Position of models
x = np.arange(len(models))

# Width of each bar
width = 0.15


# ============================================================
# CREATE FIGURE
# ============================================================

plt.figure(figsize=(12, 6))


bars1 = plt.bar(
    x - 2 * width,
    accuracy,
    width,
    label="Accuracy"
)

bars2 = plt.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

bars3 = plt.bar(
    x,
    recall,
    width,
    label="Recall"
)

bars4 = plt.bar(
    x + width,
    f1_score,
    width,
    label="F1-Score"
)

bars5 = plt.bar(
    x + 2 * width,
    roc_auc,
    width,
    label="ROC-AUC"
)


# ============================================================
# ADD VALUES ABOVE BARS
# ============================================================

def add_values(bars):

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )


add_values(bars1)
add_values(bars2)
add_values(bars3)
add_values(bars4)
add_values(bars5)


# ============================================================
# LABELS AND TITLE
# ============================================================

plt.xlabel(
    "Machine Learning Model",
    fontsize=11
)

plt.ylabel(
    "Performance (%)",
    fontsize=11
)

plt.title(
    "Performance Comparison of Random Forest and SVM",
    fontsize=14
)


plt.xticks(
    x,
    models
)


plt.ylim(
    0,
    105
)


plt.legend()


plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)


plt.tight_layout()


# ============================================================
# SAVE IMAGE
# ============================================================

plt.savefig(
    "model_comparison_rf_svm.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()