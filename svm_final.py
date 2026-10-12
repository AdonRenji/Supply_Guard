import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# SUPPLYGUARD - FAST SVM EVALUATION
# ============================================================

print("=" * 65)
print("SUPPLYGUARD - FAST SVM")
print("=" * 65)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1"
)

print("\nDataset loaded successfully!")
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
    df["order date (DateOrders)"].dt.isocalendar().week.astype(int)
)

df["Order Day of Month"] = df["order date (DateOrders)"].dt.day

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

X = df[final_features]
y = df["Late_delivery_risk"]

print("\nFinal features:", len(final_features))


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 5. SVM SUBSET
# ============================================================

X_svm, _, y_svm, _ = train_test_split(
    X_train,
    y_train,
    train_size=30000,
    random_state=42,
    stratify=y_train
)

print("SVM training records:", len(X_svm))


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 7. FAST SVM
# ============================================================

svm_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=False,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 8. TRAIN
# ============================================================

print("\n" + "=" * 65)
print("TRAINING SVM")
print("=" * 65)

start = time.time()

svm_model.fit(
    X_svm,
    y_svm
)

training_time = time.time() - start

print(
    f"\nTraining completed in {training_time:.2f} seconds."
)


# ============================================================
# 9. PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = svm_model.predict(X_test)

# Decision scores are used for ROC-AUC
y_score = svm_model.decision_function(X_test)


# ============================================================
# 10. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_score
)


# ============================================================
# 11. RESULTS
# ============================================================

print("\n" + "=" * 65)
print("SVM RESULTS")
print("=" * 65)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc * 100:.2f}%")


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 65)
print("SVM CONFUSION MATRIX")
print("=" * 65)

print(cm)

print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

print("\nCorrect predictions =", tn + tp)
print("Incorrect predictions =", fp + fn)
print("Total test records =", len(y_test))


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("PER-CLASS METRICS")
print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Late-Delivery Risk",
            "Late-Delivery Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# 14. SAVE CONFUSION MATRIX
# ============================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Late-Delivery Risk",
        "Late-Delivery Risk"
    ]
)

disp.plot(values_format="d")

plt.title("Confusion Matrix - SVM")
plt.tight_layout()

plt.savefig(
    "results/svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Model": ["SVM"],
    "Accuracy": [accuracy * 100],
    "Precision": [precision * 100],
    "Recall": [recall * 100],
    "F1 Score": [f1 * 100],
    "ROC-AUC": [roc_auc * 100],
    "Training Time (seconds)": [training_time],
    "Training Samples": [len(X_svm)],
    "Testing Samples": [len(X_test)]
})

metrics_df.to_csv(
    "results/svm_final_metrics.csv",
    index=False
)


# ============================================================
# 16. SAVE CONFUSION MATRIX VALUES
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual No Late-Delivery Risk",
        "Actual Late-Delivery Risk"
    ],
    columns=[
        "Predicted No Late-Delivery Risk",
        "Predicted Late-Delivery Risk"
    ]
)

cm_df.to_csv(
    "results/svm_confusion_matrix.csv"
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

os.makedirs("saved_model", exist_ok=True)

joblib.dump(
    svm_model,
    "saved_model/supplyguard_svm_final.joblib"
)


# ============================================================
# 18. FINISHED
# ============================================================

print("\n" + "=" * 65)
print("SVM PROCESS COMPLETED")
print("=" * 65)

print("\nFiles generated:")
print("1. results/svm_confusion_matrix.png")
print("2. results/svm_confusion_matrix.csv")
print("3. results/svm_final_metrics.csv")
print("4. saved_model/supplyguard_svm_final.joblib")

print("\nTraining time:", f"{training_time:.2f} seconds")