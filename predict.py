import os
import joblib
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_threshold.txt"
)


# ============================================================
# 2. CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Final model not found.")
    print("Please run train_final_model.py first.")
    exit()


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# 4. LOAD THRESHOLD
# ============================================================

if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read().strip())
else:
    threshold = 0.41


# ============================================================
# 5. GET USER INPUT
# ============================================================

print("\n==========================================")
print("       SUPPLYGUARD PREDICTION")
print("==========================================")
print("Enter shipment details below.\n")


shipping_mode = input(
    "Shipping Mode (Standard Class / Second Class / First Class / Same Day): "
)

scheduled_days = float(
    input("Days for shipment (scheduled): ")
)

customer_country = input("Customer Country: ")
category_name = input("Category Name: ")
quantity = float(input("Order Item Quantity: "))
market = input("Market: ")
order_region = input("Order Region: ")
order_country = input("Order Country: ")
order_state = input("Order State: ")
customer_segment = input("Customer Segment: ")
department_name = input("Department Name: ")
product_price = float(input("Order Item Product Price: "))
discount_rate = float(input("Order Item Discount Rate: "))
sales = float(input("Sales: "))

order_month = int(input("Order Month (1-12): "))
order_day = int(input("Order Day: "))
order_day_of_week = int(
    input("Order Day of Week (0=Monday, 6=Sunday): ")
)
order_hour = int(input("Order Hour (0-23): "))

is_weekend = int(
    input("Is Weekend? (0=No, 1=Yes): ")
)

order_week = int(
    input("Order Week of Year: ")
)

order_day_of_month = int(
    input("Order Day of Month: ")
)

is_month_start = int(
    input("Is Month Start? (0=No, 1=Yes): ")
)

is_month_end = int(
    input("Is Month End? (0=No, 1=Yes): ")
)

is_quarter_start = int(
    input("Is Quarter Start? (0=No, 1=Yes): ")
)

is_quarter_end = int(
    input("Is Quarter End? (0=No, 1=Yes): ")
)

is_business_hour = int(
    input("Is Business Hour? (0=No, 1=Yes): ")
)

is_night = int(
    input("Is Night? (0=No, 1=Yes): ")
)

customer_city = input("Customer City: ")
customer_state = input("Customer State: ")
shipment_type = input("Type: ")


# ============================================================
# 6. CREATE INTERACTION FEATURE
# ============================================================

interaction_shipping_hour = (
    shipping_mode + "_" + str(order_hour)
)


# ============================================================
# 7. CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([{

    "Shipping Mode": shipping_mode,
    "Days for shipment (scheduled)": scheduled_days,
    "Customer Country": customer_country,
    "Category Name": category_name,
    "Order Item Quantity": quantity,
    "Market": market,
    "Order Region": order_region,
    "Order Country": order_country,
    "Order State": order_state,
    "Customer Segment": customer_segment,
    "Department Name": department_name,
    "Order Item Product Price": product_price,
    "Order Item Discount Rate": discount_rate,
    "Sales": sales,

    "Order Month": order_month,
    "Order Day": order_day,
    "Order Day of Week": order_day_of_week,
    "Order Hour": order_hour,
    "Is Weekend": is_weekend,
    "Order Week of Year": order_week,
    "Order Day of Month": order_day_of_month,
    "Is Month Start": is_month_start,
    "Is Month End": is_month_end,
    "Is Quarter Start": is_quarter_start,
    "Is Quarter End": is_quarter_end,
    "Is Business Hour": is_business_hour,
    "Is Night": is_night,

    "Interaction_ShippingMode_OrderHour":
        interaction_shipping_hour,

    "Customer City": customer_city,
    "Type": shipment_type,
    "Customer State": customer_state

}])


# ============================================================
# 8. PREDICT PROBABILITY
# ============================================================

probability = model.predict_proba(input_data)[0][1]


# ============================================================
# 9. APPLY THRESHOLD
# ============================================================

prediction = int(probability >= threshold)


# ============================================================
# 10. DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("             PREDICTION RESULT")
print("==========================================")

print(
    f"Late Delivery Probability: {probability * 100:.2f}%"
)

print(
    f"Decision Threshold: {threshold:.2f}"
)

if prediction == 1:
    print("\n⚠️  PREDICTION: HIGH RISK OF LATE DELIVERY")
else:
    print("\n✅ PREDICTION: LOW RISK / LIKELY ON TIME")

print("==========================================\n")