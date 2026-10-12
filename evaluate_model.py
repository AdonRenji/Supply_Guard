import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# SUPPLYGUARD - FINAL MODEL EVALUATION
# ============================================================

print("=" * 70)
print("SUPPLYGUARD - FINAL MODEL EVALUATION")
print("=" * 70)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = "DataCoSupplyChainDataset.csv"

MODEL_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_threshold.txt"
)

OUTPUT_FOLDER = "results"


# Create results folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    encoding="latin1"
)

print(f"Dataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")


# ============================================================
# 3. CREATE DATE FEATURES
# ============================================================

print("\nCreating date features...")

date_column = "order date (DateOrders)"

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)


# Month
df["Order Month"] = df[date_column].dt.month

# Day
df["Order Day"] = df[date_column].dt.day

# Day of week
df["Order Day of Week"] = df[date_column].dt.dayofweek

# Hour
df["Order Hour"] = df[date_column].dt.hour

# Weekend
df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)

# Week of year
df["Order Week of Year"] = (
    df[date_column]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Day of month
df["Order Day of Month"] = df[date_column].dt.day

# Month start
df["Is Month Start"] = (
    df[date_column].dt.is_month_start
).astype(int)

# Month end
df["Is Month End"] = (
    df[date_column].dt.is_month_end
).astype(int)

# Quarter start
df["Is Quarter Start"] = (
    df[date_column].dt.is_quarter_start
).astype(int)

# Quarter end
df["Is Quarter End"] = (
    df[date_column].dt.is_quarter_end
).astype(int)

# Business hour
df["Is Business Hour"] = (
    (df["Order Hour"] >= 9)
    & (df["Order Hour"] <= 17)
    & (df["Order Day of Week"] < 5)
).astype(int)

# Night
df["Is Night"] = (
    (df["Order Hour"] < 6)
    | (df["Order Hour"] >= 22)
).astype(int)


# ============================================================
# 4. CREATE INTERACTION FEATURE
# ============================================================

print("Creating interaction feature...")

df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Order Hour"].astype(str)
)


# ============================================================
# 5. FINAL 31 FEATURES
# ============================================================

features = [

    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Customer Country",
    "Category Name",
    "Order Item Quantity",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Customer Segment",
    "Department Name",
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
    "Is Night",

    "Interaction_ShippingMode_OrderHour",

    "Customer City",
    "Type",
    "Customer State"
]

print(f"\nNumber of final features: {len(features)}")


# ============================================================
# 6. TARGET
# ============================================================

target = "Late_delivery_risk"


# ============================================================
# 7. PREPARE FEATURES AND TARGET
# ============================================================

X = df[features].copy()

y = df[target].copy()


# ============================================================
# 8. RECREATE EXACT 80/20 TEST SPLIT
# ============================================================

print("\nCreating the same 80/20 stratified split...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ============================================================
# 9. LOAD SAVED MODEL
# ============================================================

print("\nLoading final Random Forest model...")

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Final model was not found!")

    print(
        "Expected location:"
    )

    print(MODEL_PATH)

    exit()


model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# 10. LOAD CLASSIFICATION THRESHOLD
# ============================================================

if os.path.exists(THRESHOLD_PATH):

    with open(
        THRESHOLD_PATH,
        "r"
    ) as file:

        threshold = float(
            file.read().strip()
        )

else:

    threshold = 0.41


print(
    f"Classification threshold: {threshold}"
)


# ============================================================
# 11. GENERATE TEST PROBABILITIES
# ============================================================

print("\nGenerating predictions on test data...")

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 12. APPLY CLASSIFICATION THRESHOLD
# ============================================================

y_prediction = (
    y_probability >= threshold
).astype(int)


# ============================================================
# 13. CALCULATE FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_prediction
)


# ============================================================
# 15. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUPPLYGUARD MODEL EVALUATION")
print("=" * 70)

print(
    f"Accuracy:          {accuracy * 100:.2f}%"
)

print(
    f"Precision:         {precision * 100:.2f}%"
)

print(
    f"Recall:            {recall * 100:.2f}%"
)

print(
    f"F1 Score:          {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:           {roc_auc * 100:.2f}%"
)

print(
    f"Threshold:         {threshold:.2f}"
)

print("\nConfusion Matrix:")

print(cm)

print("=" * 70)


# ============================================================
# 16. SAVE FINAL METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Threshold"

    ],

    "Value": [

        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        threshold

    ]

})


metrics_path = os.path.join(
    OUTPUT_FOLDER,
    "final_evaluation_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


print(
    f"\nMetrics saved to: {metrics_path}"
)


# ============================================================
# 17. CONFUSION MATRIX VISUALIZATION
# ============================================================

print("\nCreating confusion matrix...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "On Time (0)",
        "Late (1)"
    ]

)


disp.plot(
    ax=ax,
    values_format="d"
)


ax.set_title(
    "SupplyGuard - Confusion Matrix"
)


plt.tight_layout()


confusion_path = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Confusion matrix saved to: {confusion_path}"
)


# ============================================================
# 18. ROC CURVE
# ============================================================

print("\nCreating ROC curve...")

fpr, tpr, thresholds_roc = roc_curve(
    y_test,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)


# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "SupplyGuard - ROC Curve"
)


plt.legend(
    loc="lower right"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


roc_path = os.path.join(
    OUTPUT_FOLDER,
    "roc_curve.png"
)


plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"ROC curve saved to: {roc_path}"
)


# ============================================================
# 19. FIND MODEL PIPELINE COMPONENTS
# ============================================================

print("\nExtracting feature importance...")

print(
    "\nSaved model pipeline steps:"
)

print(
    model.named_steps.keys()
)


# ------------------------------------------------------------
# Find the Random Forest automatically
# ------------------------------------------------------------

random_forest = None

preprocessor = None


for step_name, step_object in model.named_steps.items():

    # Check for Random Forest
    if hasattr(
        step_object,
        "feature_importances_"
    ):

        random_forest = step_object

        print(
            f"Random Forest found in step: {step_name}"
        )


    # Check for preprocessing object
    if hasattr(
        step_object,
        "get_feature_names_out"
    ):

        preprocessor = step_object

        print(
            f"Preprocessor found in step: {step_name}"
        )


# ============================================================
# 20. CHECK COMPONENTS
# ============================================================

if random_forest is None:

    print(
        "\nERROR: Random Forest step could not be found."
    )

    print(
        "Feature importance cannot be generated."
    )

    exit()


if preprocessor is None:

    print(
        "\nERROR: Preprocessor step could not be found."
    )

    print(
        "Feature importance cannot be generated."
    )

    exit()


print(
    "\nRandom Forest step found successfully."
)

print(
    "Preprocessor step found successfully."
)


# ============================================================
# 21. GET TRANSFORMED FEATURE NAMES
# ============================================================

print(
    "\nGetting transformed feature names..."
)


feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# 22. GET RANDOM FOREST IMPORTANCES
# ============================================================

importances = (
    random_forest.feature_importances_
)


print(
    f"Number of transformed features: "
    f"{len(feature_names)}"
)

print(
    f"Number of importance values: "
    f"{len(importances)}"
)


# ============================================================
# 23. CREATE FEATURE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importances

})


importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)


# ============================================================
# 24. SAVE COMPLETE FEATURE IMPORTANCE
# ============================================================

importance_csv_path = os.path.join(

    OUTPUT_FOLDER,

    "feature_importance.csv"

)


importance_df.to_csv(

    importance_csv_path,

    index=False

)


print(
    f"\nFeature importance saved to: "
    f"{importance_csv_path}"
)


# ============================================================
# 25. PRINT TOP 20 FEATURES
# ============================================================

print("\nTop 20 Features:")

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 26. CREATE TOP 20 FEATURE IMPORTANCE PLOT
# ============================================================

print(
    "\nCreating feature importance plot..."
)


top_features = (
    importance_df
    .head(20)
    .copy()
)


plt.figure(
    figsize=(10, 8)
)


plt.barh(

    top_features["Feature"][::-1],

    top_features["Importance"][::-1]

)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "SupplyGuard - Top 20 Feature Importances"
)


plt.tight_layout()


importance_plot_path = os.path.join(

    OUTPUT_FOLDER,

    "feature_importance_top20.png"

)


plt.savefig(

    importance_plot_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    f"Feature importance plot saved to: "
    f"{importance_plot_path}"
)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nFinal Metrics:")

print(
    f"Accuracy:   {accuracy * 100:.2f}%"
)

print(
    f"Precision:  {precision * 100:.2f}%"
)

print(
    f"Recall:     {recall * 100:.2f}%"
)

print(
    f"F1 Score:   {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:    {roc_auc * 100:.2f}%"
)


print("\nGenerated files:")

print(
    "1. results/final_evaluation_metrics.csv"
)

print(
    "2. results/confusion_matrix.png"
)

print(
    "3. results/roc_curve.png"
)

print(
    "4. results/feature_importance.csv"
)

print(
    "5. results/feature_importance_top20.png"
)

print("\nModel was NOT retrained.")

print("=" * 70)