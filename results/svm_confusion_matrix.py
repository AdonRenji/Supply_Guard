import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["Order Month"] = df["order date (DateOrders)"].dt.month
df["Order Day"] = df["order date (DateOrders)"].dt.day
df["Order Day of Week"] = df["order date (DateOrders)"].dt.dayofweek
df["Order Hour"] = df["order date (DateOrders)"].dt.hour

df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)

df["Order Week of Year"] = (
    df["order date (DateOrders)"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["Order Day of Month"] = (
    df["order date (DateOrders)"].dt.day
)

df["Is Month Start"] = (
    df["order date (DateOrders)"].dt.is_month_start
).astype(int)

df["Is Month End"] = (
    df["order date (DateOrders)"].dt.is_month_end
).astype(int)

df["Is Quarter Start"] = (
    df["order date (DateOrders)"].dt.is_quarter_start
).astype(int)

df["Is Quarter End"] = (
    df["order date (DateOrders)"].dt.is_quarter_end
).astype(int)

df["Is Business Hour"] = (
    (df["Order Hour"] >= 9) &
    (df["Order Hour"] < 18)
).astype(int)

df["Is Night"] = (
    (df["Order Hour"] < 6) |
    (df["Order Hour"] >= 22)
).astype(int)

# Interaction feature
df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Order Hour"].astype(str)
)


# ============================================================
# 3. FINAL 31 FEATURES
# ============================================================

categorical_features = [
    "Shipping Mode",
    "Customer Country",
    "Category Name",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Customer Segment",
    "Department Name",
    "Interaction_ShippingMode_OrderHour",
    "Customer City",
    "Type",
    "Customer State"
]

numerical_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "Sales",
    "Order Month",
    "Order Day",
    "Order Day of Week",
    "Order Hour",
    "Is Weekend",
    "Order Week of Year",
    "Order Day of Month",
    "Is Month Start",
    "Is Month End",
    "Is Quarter Start",
    "Is Quarter End",
    "Is Business Hour",
    "Is Night"
]

final_features = (
    categorical_features +
    numerical_features
)

print("\nFinal features:", len(final_features))


# ============================================================
# 4. INPUT AND TARGET
# ============================================================

X = df[final_features].copy()
y = df["Late_delivery_risk"].copy()


# ============================================================
# 5. 80:20 STRATIFIED TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

# Fit ONLY on training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_transformed = preprocessor.transform(X_test)

print(
    "\nTransformed training shape:",
    X_train_transformed.shape
)

print(
    "Transformed testing shape:",
    X_test_transformed.shape
)


# ============================================================
# 7. STRATIFIED 30,000-RECORD SVM TRAINING SUBSET
# ============================================================

X_train_svm, _, y_train_svm, _ = train_test_split(
    X_train,
    y_train,
    train_size=30000,
    random_state=42,
    stratify=y_train
)

# IMPORTANT:
# Use the already-fitted preprocessor.
X_train_svm_transformed = preprocessor.transform(
    X_train_svm
)

print("\nSVM training records:", len(X_train_svm))


# ============================================================
# 8. CREATE SVM
# ============================================================

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)


# ============================================================
# 9. TRAIN SVM
# ============================================================

print("\nTraining SVM...")
print("This may take some time.")

svm_model.fit(
    X_train_svm_transformed,
    y_train_svm
)

print("SVM training completed!")


# ============================================================
# 10. TEST PREDICTIONS
# ============================================================

svm_preds = svm_model.predict(
    X_test_transformed
)

svm_proba = svm_model.predict_proba(
    X_test_transformed
)[:, 1]


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    svm_preds
)

tn, fp, fn, tp = cm.ravel()

print("\n========================================")
print("SVM CONFUSION MATRIX")
print("========================================")

print(cm)

print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

print("\nCorrect predictions =", tn + tp)
print("Incorrect predictions =", fp + fn)
print("Total test records =", tn + fp + fn + tp)


# ============================================================
# 12. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    svm_preds
)

precision = precision_score(
    y_test,
    svm_preds
)

recall = recall_score(
    y_test,
    svm_preds
)

f1 = f1_score(
    y_test,
    svm_preds
)

roc_auc = roc_auc_score(
    y_test,
    svm_proba
)

print("\n========================================")
print("SVM EVALUATION RESULTS")
print("========================================")

print(f"Accuracy :  {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}%")
print(f"Recall   :  {recall * 100:.2f}%")
print(f"F1-Score :  {f1 * 100:.2f}%")
print(f"ROC-AUC  :  {roc_auc * 100:.2f}%")


# ============================================================
# 13. PER-CLASS METRICS
# ============================================================

print("\n========================================")
print("PER-CLASS METRICS")
print("========================================")

print(
    "\nClassification report:\n"
)

from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        svm_preds,
        target_names=[
            "No Late-Delivery Risk",
            "Late-Delivery Risk"
        ],
        digits=4
    )
)


# ============================================================
# 14. PLOT CONFUSION MATRIX
# ============================================================

labels = [
    "No Late-Delivery Risk",
    "Late-Delivery Risk"
]

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="white"
)

plt.title(
    "SVM - Confusion Matrix",
    fontsize=14
)

plt.xlabel(
    "Predicted",
    fontsize=12
)

plt.ylabel(
    "Actual",
    fontsize=12
)

plt.xticks(rotation=0)
plt.yticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nConfusion matrix saved as: "
    "svm_confusion_matrix.png"
)