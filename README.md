give

Create README.md in the project root, next to requirements.txt, and paste this:

# SupplyGuard

## AI-Powered Shipment Delay Prediction and Supplier Recommendation System

SupplyGuard is a machine learning-based mini project designed to predict the risk of shipment delays using historical supply chain data.

The system uses shipment, order, customer, product, geographic, and calendar-related information to predict whether a shipment is likely to be delivered late. A Random Forest classification model is used as the primary machine learning algorithm.

The project focuses on the machine learning implementation and evaluation of shipment delay prediction.

---

## 1. Project Objective

The main objective of SupplyGuard is to develop a machine learning model that can predict shipment late-delivery risk before shipment completion.

The system:

- Analyses historical supply chain data.
- Performs feature selection and feature engineering.
- Preprocesses numerical and categorical features.
- Uses One-Hot Encoding for categorical variables.
- Trains a Random Forest classification model.
- Produces a probability of late delivery.
- Applies a classification threshold to determine the predicted risk.
- Evaluates the model using standard classification metrics.
- Provides visual evaluation outputs such as a confusion matrix, ROC curve, and feature importance.

---

## 2. Dataset

The project uses the **DataCo Smart Supply Chain Dataset**.

Dataset characteristics:

- Total records: **180,519**
- Original columns: **53**
- Target variable: `Late_delivery_risk`

Target interpretation:

```text
0 = On-time delivery
1 = Late delivery

Target distribution:

On-time shipments: 81,542
Late shipments:    98,977

The original dataset contains shipment, order, customer, product, geographic, and delivery-related information.

Outcome-related variables that could cause data leakage were not used as prediction features.

3. Feature Selection

The final model uses 31 features.

Operational features
Shipping Mode
Days for shipment (scheduled)
Customer Country
Category Name
Order Item Quantity
Market
Order Region
Order Country
Order State
Customer Segment
Department Name
Order Item Product Price
Order Item Discount Rate
Sales
Calendar features
Order Month
Order Day
Order Day of Week
Order Hour
Is Weekend
Order Week of Year
Order Day of Month
Is Month Start
Is Month End
Is Quarter Start
Is Quarter End
Is Business Hour
Is Night
Engineered interaction feature
Interaction_ShippingMode_OrderHour
Additional customer/transaction features
Customer City
Type
Customer State
4. Feature Engineering

The original order date was used to derive calendar-related features.

The following features were generated:

Month
Day
Day of week
Hour
Weekend indicator
Week of year
Day of month
Month start indicator
Month end indicator
Quarter start indicator
Quarter end indicator
Business-hour indicator
Night-time indicator

An interaction feature was also created:

Interaction_ShippingMode_OrderHour

This represents the combination of shipping mode and order hour.

5. Data Preprocessing

The model uses a preprocessing pipeline.

Numerical features

Numerical features are processed using median imputation.

Categorical features

Categorical features are processed using:

Most-frequent value imputation.
One-Hot Encoding.

The One-Hot Encoder uses:

handle_unknown = ignore

This allows the model to handle previously unseen categorical values during prediction.

The preprocessing steps are included inside the saved machine learning pipeline.

6. Train-Test Split

The dataset was divided using an 80/20 stratified split.

Training records: 144,415
Testing records:   36,104

Stratification was used to preserve the target-class distribution between the training and testing sets.

The test set was kept separate for final model evaluation.

7. Machine Learning Algorithm

The final model is a Random Forest Classifier.

Configuration:

Algorithm: Random Forest
Number of trees: 500
Maximum depth: None
Minimum samples per leaf: 2
Class weight: None
Random state: 42
Number of CPU jobs: -1

The model was selected after comparing different feature configurations, tree counts, feature engineering approaches, and other experimental models.

8. Classification Threshold

The final model uses a classification threshold of:

0.41

The Random Forest first produces the probability of late delivery.

The probability is then converted into a binary prediction:

Probability >= 0.41  → Late delivery risk
Probability < 0.41   → On-time prediction

The threshold was selected using validation data rather than the final test set.

9. Final Model Performance

The final model was evaluated on the held-out test set.

Metric	Result
Accuracy	86.40%
Precision	86.18%
Recall	89.56%
F1 Score	87.84%
ROC-AUC	93.13%
Confusion Matrix
                  Predicted
                On Time   Late
Actual On Time    13464   2844
Actual Late        2066  17730

The model correctly classified:

13,464 on-time shipments.
17,730 late shipments.
10. Model Evaluation Outputs

The evaluate_model.py script generates the following files:

results/
├── final_evaluation_metrics.csv
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.csv
└── feature_importance_top20.png
Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for both classes.

ROC Curve

The ROC curve evaluates the model's ability to distinguish between on-time and late shipments.

The final ROC-AUC is:

0.9313
Feature Importance

Feature importance is extracted from the Random Forest model.

The most important transformed features include:

Days for shipment (scheduled)
Shipping Mode - Standard Class
Shipping Mode - First Class
Shipping Mode - Second Class
Order Hour
Order Week of Year
Order Day of Month
Order Day
Shipping Mode - Same Day
Order Day of Week
11. Project Files

The main project structure is:

adon suplyguard/
│
├── DataCoSupplyChainDataset.csv
│
├── train_final_model.py
├── predict.py
├── test_prediction.py
├── evaluate_model.py
├── requirements.txt
├── README.md
│
├── saved_model/
│   ├── supplyguard_final_model.joblib
│   ├── supplyguard_final_features.txt
│   ├── supplyguard_final_threshold.txt
│   ├── supplyguard_final_metrics.csv
│   └── supplyguard_final_config.txt
│
└── results/
    ├── final_evaluation_metrics.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.csv
    └── feature_importance_top20.png
12. Python Scripts
train_final_model.py

Trains the final Random Forest model using the selected 31 features and saves the trained pipeline.

predict.py

Allows a user to manually enter shipment information and obtain the predicted late-delivery probability and risk classification.

test_prediction.py

Verifies the saved model using an actual record from the held-out test set and compares the predicted class with the actual Late_delivery_risk value.

evaluate_model.py

Performs final evaluation of the saved model and generates:

Classification metrics
Confusion matrix
ROC curve
Feature importance table
Feature importance visualization
13. Installation

Install the required Python packages using:

pip install -r requirements.txt

Required packages include:

pandas
numpy
scikit-learn
matplotlib
joblib
14. How to Run
Step 1: Train the model
python train_final_model.py

This creates the saved model inside:

saved_model/
Step 2: Perform a manual prediction
python predict.py

Enter the required shipment information when prompted.

Step 3: Verify a test record
python test_prediction.py

This checks the model against an actual held-out dataset record.

Step 4: Evaluate the final model
python evaluate_model.py

This generates the final evaluation metrics and visualizations.

15. Leakage Prevention

The project considers whether a feature would be available at the time the prediction is made.

Outcome-related variables such as actual delivery information were excluded from the final prediction features because they would not be appropriate for an early shipment-risk prediction.

The final model therefore focuses on information available from the order, shipment planning, customer, product, geographic, and calendar context.

16. Limitations

The current project has several limitations:

The model is trained using historical supply chain data.
External real-time factors such as traffic, weather, port congestion, and GPS information are not included.
The model depends on the quality and characteristics of the available historical dataset.
Supplier recommendation is not learned directly by the Random Forest model.
Real-time deployment and continuous model updating are outside the current machine learning implementation.
17. Future Scope

Possible future improvements include:

Real-time shipment tracking.
GPS-based shipment monitoring.
Weather and traffic information.
Port and transportation congestion data.
Additional supplier performance information.
Automated alerts for high-risk shipments.
Periodic model retraining with new shipment data.
Expansion to real-time supply chain risk monitoring.
18. Project Summary

SupplyGuard demonstrates the application of machine learning to shipment late-delivery risk prediction.

The final Random Forest model uses 31 selected and engineered features, One-Hot Encoding for categorical variables, and a probability threshold of 0.41.

On the held-out test set, the model achieved:

Accuracy:   86.40%
Precision:  86.18%
Recall:     89.56%
F1 Score:   87.84%
ROC-AUC:    93.13%

The project provides a complete machine learning workflow from dataset preparation and feature engineering to model training, prediction, verification, and final evaluation.