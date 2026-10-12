
import os
from datetime import datetime, date

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "DataCoSupplyChainDataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "saved_model", "supplyguard_final_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR, "saved_model", "supplyguard_final_threshold.txt"
)

# --------------------------------------------------
# Load model and dataset
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

df = pd.read_csv(
    DATASET_PATH,
    encoding="latin1"
)

threshold = 0.41

if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold = float(file.read().strip())

# --------------------------------------------------
# Dropdown options
# --------------------------------------------------

categorical_fields = {
    "shipping_mode": "Shipping Mode",
    "customer_country": "Customer Country",
    "category_name": "Category Name",
    "market": "Market",
    "order_region": "Order Region",
    "order_country": "Order Country",
    "order_state": "Order State",
    "customer_segment": "Customer Segment",
    "department_name": "Department Name",
    "customer_city": "Customer City",
    "type": "Type",
    "customer_state": "Customer State",
}

options = {
    field: sorted(
        df[column].dropna().unique().tolist(),
        key=lambda value: str(value).lower()
    )
    for field, column in categorical_fields.items()
}

days_options = sorted(
    int(value)
    for value in df["Days for shipment (scheduled)"].dropna().unique()
)

quantity_options = sorted(
    int(value)
    for value in df["Order Item Quantity"].dropna().unique()
)

# --------------------------------------------------
# Date range
# Allow dates from 1 January 2015 through today
# --------------------------------------------------

min_date = "2015-01-01"
max_date = date.today().strftime("%Y-%m-%d")

print("SupplyGuard model loaded successfully.")
print("Available order dates:", min_date, "to", max_date)


# --------------------------------------------------
# Prediction page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None
    form_data = {}

    if request.method == "POST":

        form_data = request.form.to_dict()

        try:
            # ------------------------------
            # Read form inputs
            # ------------------------------

            shipping_mode = form_data["shipping_mode"]

            order_date = datetime.strptime(
                form_data["order_date"], "%Y-%m-%d"
            )

            order_time = datetime.strptime(
                form_data["order_time"], "%H:%M"
            ).time()

            # Validate date range on the server too
            selected_date = order_date.date()

            if selected_date < date(2015, 1, 1):
                raise ValueError(
                    "Order date cannot be before 1 January 2015."
                )

            if selected_date > date.today():
                raise ValueError(
                    "Order date cannot be later than today."
                )

            order_datetime = pd.Timestamp(
                datetime.combine(
                    order_date.date(),
                    order_time
                )
            )

            hour = order_datetime.hour
            day_of_week = order_datetime.dayofweek

            # ------------------------------
            # Prepare model input
            # Keep the feature names consistent
            # with the trained model.
            # ------------------------------

            row = {
                "Shipping Mode": shipping_mode,

                "Days for shipment (scheduled)": int(
                    form_data["days_scheduled"]
                ),

                "Customer Country": form_data["customer_country"],

                "Category Name": form_data["category_name"],

                "Order Item Quantity": int(
                    form_data["quantity"]
                ),

                "Market": form_data["market"],

                "Order Region": form_data["order_region"],

                "Order Country": form_data["order_country"],

                "Order State": form_data["order_state"],

                "Customer Segment": form_data["customer_segment"],

                "Department Name": form_data["department_name"],

                "Order Item Product Price": float(
                    form_data["product_price"]
                ),

                "Order Item Discount Rate": float(
                    form_data["discount_rate"]
                ),

                "Sales": float(form_data["sales"]),

                "Order Month": order_datetime.month,

                "Order Day": order_datetime.day,

                "Order Day of Week": day_of_week,

                "Order Hour": hour,

                "Is Weekend": int(day_of_week >= 5),

                "Order Week of Year": int(
                    order_datetime.isocalendar().week
                ),

                "Order Day of Month": order_datetime.day,

                "Is Month Start": int(
                    order_datetime.is_month_start
                ),

                "Is Month End": int(
                    order_datetime.is_month_end
                ),

                "Is Quarter Start": int(
                    order_datetime.is_quarter_start
                ),

                "Is Quarter End": int(
                    order_datetime.is_quarter_end
                ),

                "Is Business Hour": int(
                    day_of_week < 5 and 9 <= hour < 17
                ),

                "Is Night": int(
                    hour < 6 or hour >= 22
                ),

                "Interaction_ShippingMode_OrderHour":
                    f"{shipping_mode}_{hour}",

                "Customer City": form_data["customer_city"],

                "Type": form_data["type"],

                "Customer State": form_data["customer_state"],
            }

            # ------------------------------
            # Arrange features in trained order
            # ------------------------------

            final_features = [
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
                "Customer State",
            ]

            input_data = pd.DataFrame(
                [row],
                columns=final_features
            )

            # ------------------------------
            # Predict shipment risk
            # ------------------------------

            probability = float(
                model.predict_proba(input_data)[0][1]
            )

            prediction = int(probability >= threshold)

            result = {
                "probability": round(probability * 100, 2),
                "threshold": round(threshold * 100, 2),
                "prediction": prediction,
                "label": (
                    "HIGH RISK"
                    if prediction == 1
                    else "LOW RISK"
                ),
            }

        except (ValueError, KeyError, TypeError) as exc:

            error = (
                "Please check your entries and try again. "
                f"Details: {exc}"
            )

    # ------------------------------
    # Render the page
    # ------------------------------

    return render_template(
        "index.html",
        options=options,
        days_options=days_options,
        quantity_options=quantity_options,
        min_date=min_date,
        max_date=max_date,
        result=result,
        error=error,
        form_data=form_data,
    )


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
