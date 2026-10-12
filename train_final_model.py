# ============================================================
# SupplyGuard - Final Random Forest Model
# ============================================================
#
# Final configuration:
#   Algorithm            : Random Forest
#   Number of trees      : 500
#   Maximum depth        : None
#   Minimum samples leaf : 2
#   Class weight         : None
#   Random state         : 42
#   Classification       : threshold = 0.41
#
# Dataset:
#   DataCo Smart Supply Chain Dataset
#
# Features:
#   31 selected features
#
# ============================================================


import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "DataCoSupplyChainDataset.csv"

MODEL_FOLDER = "saved_model"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "supplyguard_final_model.joblib"
)

FEATURES_PATH = os.path.join(
    MODEL_FOLDER,
    "supplyguard_final_features.txt"
)

THRESHOLD_PATH = os.path.join(
    MODEL_FOLDER,
    "supplyguard_final_threshold.txt"
)

METRICS_PATH = os.path.join(
    MODEL_FOLDER,
    "supplyguard_final_metrics.csv"
)

CONFIG_PATH = os.path.join(
    MODEL_FOLDER,
    "supplyguard_final_config.txt"
)


# Final classification threshold
THRESHOLD = 0.41

# Reproducibility
RANDOM_STATE = 42


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 65)
print("SUPPLYGUARD - FINAL MODEL TRAINING")
print("=" * 65)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    encoding="latin1"
)

print("Dataset loaded successfully!")

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 4. CREATE DATE FEATURES
# ============================================================

print("\nCreating date features...")

df["DateOrders"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)


# Month
df["Order Month"] = (
    df["DateOrders"].dt.month
)


# Day
df["Order Day"] = (
    df["DateOrders"].dt.day
)


# Day of week
# Monday = 0
# Sunday = 6
df["Order Day of Week"] = (
    df["DateOrders"].dt.dayofweek
)


# Hour
df["Order Hour"] = (
    df["DateOrders"].dt.hour
)


# Weekend flag
df["Is Weekend"] = (
    df["Order Day of Week"] >= 5
).astype(int)


# Week of year
df["Order Week of Year"] = (
    df["DateOrders"]
    .dt.isocalendar()
    .week
    .astype(int)
)


# Day of month
df["Order Day of Month"] = (
    df["DateOrders"].dt.day
)


# Month start
df["Is Month Start"] = (
    df["DateOrders"].dt.is_month_start
).astype(int)


# Month end
df["Is Month End"] = (
    df["DateOrders"].dt.is_month_end
).astype(int)


# Quarter start
df["Is Quarter Start"] = (
    df["DateOrders"].dt.is_quarter_start
).astype(int)


# Quarter end
df["Is Quarter End"] = (
    df["DateOrders"].dt.is_quarter_end
).astype(int)


# Business hour
df["Is Business Hour"] = (
    (df["Order Day of Week"] < 5)
    &
    (df["Order Hour"] >= 9)
    &
    (df["Order Hour"] <= 17)
).astype(int)


# Night
df["Is Night"] = (
    (df["Order Hour"] < 6)
    |
    (df["Order Hour"] >= 22)
).astype(int)


# ============================================================
# 5. CREATE INTERACTION FEATURE
# ============================================================

print("Creating interaction feature...")

df["Interaction_ShippingMode_OrderHour"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Order Hour"].astype(str)
)


# ============================================================
# 6. FINAL 31 FEATURES
# ============================================================

features = [

    # Original operational features
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

    # Calendar features
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

    # Interaction feature
    "Interaction_ShippingMode_OrderHour",

    # Additional selected features
    "Customer City",
    "Type",
    "Customer State"
]


target = "Late_delivery_risk"


print("\nNumber of final features:", len(features))

print("\nFinal features:")

for i, feature in enumerate(features, start=1):
    print(f"{i:2}. {feature}")


# ============================================================
# 7. CREATE X AND y
# ============================================================

X = df[features].copy()

y = df[target].copy()


# ============================================================
# 8. DEFINE FEATURE TYPES
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


print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# ============================================================
# 9. CHECK FEATURE COUNT
# ============================================================

if len(features) != 31:

    raise ValueError(
        "ERROR: Expected exactly 31 features."
    )


if (
    len(categorical_features)
    + len(numerical_features)
    != len(features)
):

    raise ValueError(
        "ERROR: Feature type counts do not match."
    )


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

print("\nSplitting dataset using 80/20 stratified split...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nData split:")

print(
    "Training:",
    len(X_train)
)

print(
    "Testing: ",
    len(X_test)
)


# ============================================================
# 11. PREPROCESSING
# ============================================================

print("\nCreating preprocessing pipeline...")


# Numerical preprocessing
#
# Missing numerical values are replaced
# with the median value.

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# Categorical preprocessing
#
# Missing categorical values are replaced
# with the most frequent value.
#
# OneHotEncoder converts categorical values
# into numerical columns.
#
# handle_unknown="ignore" allows the model
# to handle unseen categories.

categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            numeric_transformer,
            numerical_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 12. CREATE FINAL RANDOM FOREST
# ============================================================

print("\nCreating final Random Forest model...")

random_forest = RandomForestClassifier(

    # Final selected configuration
    n_estimators=500,

    max_depth=None,

    min_samples_leaf=2,

    class_weight=None,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


# ============================================================
# 13. COMPLETE MODEL PIPELINE
# ============================================================

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            random_forest
        )
    ]
)


print("\nFinal Random Forest configuration:")

print("Algorithm: Random Forest")
print("Number of trees: 500")
print("Maximum depth: None")
print("Minimum samples per leaf: 2")
print("Class weight: None")
print("Random state: 42")
print("Number of CPU threads: All available")
print("Classification threshold:", THRESHOLD)


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\n" + "=" * 65)
print("TRAINING FINAL MODEL")
print("=" * 65)

print("\nTraining model...")
print("This may take several minutes.")

start_time = time.time()


model.fit(
    X_train,
    y_train
)


training_time = (
    time.time() - start_time
)


print("\nTraining completed!")

print(
    "Training time: {:.2f} seconds".format(
        training_time
    )
)


# ============================================================
# 15. GENERATE TEST PROBABILITIES
# ============================================================

print("\nGenerating test probabilities...")

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 16. APPLY FINAL THRESHOLD
# ============================================================

print(
    "\nApplying classification threshold:",
    THRESHOLD
)


y_pred = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# 17. CALCULATE METRICS
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
    probabilities
)


cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 18. DISPLAY FINAL RESULTS
# ============================================================

print("\n")

print("=" * 65)
print("SUPPLYGUARD FINAL RANDOM FOREST RESULTS")
print("=" * 65)

print(
    "Accuracy:          {:.2f}%".format(
        accuracy * 100
    )
)

print(
    "Precision:         {:.2f}%".format(
        precision * 100
    )
)

print(
    "Recall:            {:.2f}%".format(
        recall * 100
    )
)

print(
    "F1 Score:          {:.2f}%".format(
        f1 * 100
    )
)

print(
    "ROC-AUC:           {:.2f}%".format(
        roc_auc * 100
    )
)

print(
    "Threshold:         {:.2f}".format(
        THRESHOLD
    )
)

print(
    "Training time:     {:.2f} seconds".format(
        training_time
    )
)


print("\nConfusion Matrix:")

print(cm)

print("=" * 65)


# ============================================================
# 19. SAVE TRAINED MODEL
# ============================================================

print("\nSaving final trained model...")

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "Model saved to:",
    MODEL_PATH
)


# ============================================================
# 20. SAVE FEATURE LIST
# ============================================================

with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8"
) as file:

    for feature in features:

        file.write(
            feature + "\n"
        )


print(
    "Features saved to:",
    FEATURES_PATH
)


# ============================================================
# 21. SAVE THRESHOLD
# ============================================================

with open(
    THRESHOLD_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        str(THRESHOLD)
    )


print(
    "Threshold saved to:",
    THRESHOLD_PATH
)


# ============================================================
# 22. SAVE FINAL METRICS
# ============================================================

metrics = pd.DataFrame([

    {

        "Model": "Random Forest",

        "Estimators": 500,

        "Max Depth": "None",

        "Min Samples Leaf": 2,

        "Class Weight": "None",

        "Random State": 42,

        "Features": len(features),

        "Threshold": THRESHOLD,

        "Training Samples": len(X_train),

        "Testing Samples": len(X_test),

        "Accuracy": accuracy * 100,

        "Precision": precision * 100,

        "Recall": recall * 100,

        "F1 Score": f1 * 100,

        "ROC-AUC": roc_auc * 100,

        "Training Time Seconds": training_time

    }

])


metrics.to_csv(
    METRICS_PATH,
    index=False
)


print(
    "Metrics saved to:",
    METRICS_PATH
)


# ============================================================
# 23. SAVE MODEL CONFIGURATION
# ============================================================

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SUPPLYGUARD FINAL MODEL CONFIGURATION\n"
    )

    file.write(
        "=====================================\n\n"
    )

    file.write(
        "Algorithm: Random Forest\n"
    )

    file.write(
        "Number of trees: 500\n"
    )

    file.write(
        "Maximum depth: None\n"
    )

    file.write(
        "Minimum samples per leaf: 2\n"
    )

    file.write(
        "Class weight: None\n"
    )

    file.write(
        "Random state: 42\n"
    )

    file.write(
        "Classification threshold: 0.41\n"
    )

    file.write(
        f"Number of features: {len(features)}\n"
    )

    file.write(
        f"Training samples: {len(X_train)}\n"
    )

    file.write(
        f"Testing samples: {len(X_test)}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: {f1 * 100:.2f}%\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc * 100:.2f}%\n"
    )

    file.write(
        f"Training time: {training_time:.2f} seconds\n"
    )


print(
    "Configuration saved to:",
    CONFIG_PATH
)


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n")

print("=" * 65)
print("FINAL SUPPLYGUARD MODEL CREATED SUCCESSFULLY!")
print("=" * 65)

print("\nGenerated files:")

print(
    "1.",
    MODEL_PATH
)

print(
    "2.",
    FEATURES_PATH
)

print(
    "3.",
    THRESHOLD_PATH
)

print(
    "4.",
    METRICS_PATH
)

print(
    "5.",
    CONFIG_PATH
)

print("\nFinal model is ready for prediction.")

print("=" * 65)