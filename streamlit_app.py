import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SupplyGuard",
    page_icon="🚚",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    "saved_model",
    "supplyguard_final_threshold.txt"
)

DATASET_PATH = "DataCoSupplyChainDataset.csv"


# ============================================================
# LOAD MODEL - CACHED
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


if model is None:

    st.error(
        "Final model not found. "
        "Please make sure the saved_model folder exists."
    )

    st.stop()


# ============================================================
# LOAD THRESHOLD - CACHED
# ============================================================

@st.cache_data
def load_threshold():

    if os.path.exists(THRESHOLD_PATH):

        with open(THRESHOLD_PATH, "r") as file:
            return float(file.read().strip())

    return 0.41


threshold = load_threshold()


# ============================================================
# LOAD DATASET - CACHED
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATASET_PATH,
        encoding="latin1"
    )


try:

    df = load_dataset()

except Exception as e:

    st.error(
        f"Unable to load DataCo dataset: {e}"
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def unique_values(column):

    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


def numeric_values(column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna().unique()

    return sorted(values.tolist())


# ============================================================
# DATASET VALUES
# ============================================================

shipping_modes = unique_values(
    "Shipping Mode"
)

customer_countries = unique_values(
    "Customer Country"
)

category_names = unique_values(
    "Category Name"
)

markets = unique_values(
    "Market"
)

order_regions = unique_values(
    "Order Region"
)

order_countries = unique_values(
    "Order Country"
)

order_states = unique_values(
    "Order State"
)

customer_segments = unique_values(
    "Customer Segment"
)

department_names = unique_values(
    "Department Name"
)

customer_cities = unique_values(
    "Customer City"
)

shipment_types = unique_values(
    "Type"
)

customer_states = unique_values(
    "Customer State"
)


# ============================================================
# NUMERICAL VALUES
# ============================================================

quantity_values = numeric_values(
    "Order Item Quantity"
)

scheduled_day_values = numeric_values(
    "Days for shipment (scheduled)"
)

quantity_values = [
    int(x) for x in quantity_values
]

scheduled_day_values = [
    int(x) for x in scheduled_day_values
]


# ============================================================
# ORDER DATE RANGE
# ============================================================

# Convert the original dataset order-date column.
# DataCo dates are stored in month/day/year format.

dataset_dates = pd.to_datetime(
    df["order date (DateOrders)"],
    format="%m/%d/%Y %H:%M",
    errors="coerce"
).dropna()


if len(dataset_dates) == 0:

    st.error(
        "Unable to read valid order dates from the dataset."
    )

    st.stop()


min_order_date = dataset_dates.min().date()
max_order_date = dataset_dates.max().date()


# ============================================================
# TITLE
# ============================================================

st.title("🚚 SupplyGuard")

st.subheader(
    "AI-Powered Shipment Delay Prediction System"
)

st.write(
    "Enter shipment information to predict "
    "the risk of late delivery."
)

st.divider()


# ============================================================
# FORM
# ============================================================

with st.form("supplyguard_prediction_form"):


    # ========================================================
    # SHIPMENT INFORMATION
    # ========================================================

    st.header("📦 Shipment Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        shipping_mode = st.selectbox(
            "Shipping Mode",
            shipping_modes
        )


    with col2:

        scheduled_days = st.selectbox(
            "Days for Shipment (Scheduled)",
            scheduled_day_values
        )


    with col3:

        market = st.selectbox(
            "Market",
            markets
        )


    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.header("👤 Customer Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        customer_country = st.selectbox(
            "Customer Country",
            customer_countries
        )


    with col2:

        customer_state = st.selectbox(
            "Customer State",
            customer_states
        )


    with col3:

        customer_city = st.selectbox(
            "Customer City",
            customer_cities
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        customer_segment = st.selectbox(
            "Customer Segment",
            customer_segments
        )


    with col2:

        order_region = st.selectbox(
            "Order Region",
            order_regions
        )


    with col3:

        order_country = st.selectbox(
            "Order Country",
            order_countries
        )


    order_state = st.selectbox(
        "Order State",
        order_states
    )


    # ========================================================
    # PRODUCT INFORMATION
    # ========================================================

    st.header("🛍️ Product Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        category_name = st.selectbox(
            "Category Name",
            category_names
        )


    with col2:

        department_name = st.selectbox(
            "Department Name",
            department_names
        )


    with col3:

        shipment_type = st.selectbox(
            "Type",
            shipment_types
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        quantity = st.selectbox(
            "Order Item Quantity",
            quantity_values
        )


    with col2:

        product_price = st.number_input(
            "Order Item Product Price",
            min_value=0.0,
            max_value=float(
                df["Order Item Product Price"].max()
            ),
            value=float(
                df["Order Item Product Price"].median()
            ),
            step=1.0
        )


    with col3:

        discount_rate = st.number_input(
            "Order Item Discount Rate",
            min_value=float(
                df["Order Item Discount Rate"].min()
            ),
            max_value=float(
                df["Order Item Discount Rate"].max()
            ),
            value=float(
                df["Order Item Discount Rate"].median()
            ),
            step=0.01
        )


    sales = st.number_input(
        "Sales",
        min_value=0.0,
        max_value=float(
            df["Sales"].max()
        ),
        value=float(
            df["Sales"].median()
        ),
        step=1.0
    )


    # ========================================================
    # ORDER DATE AND TIME
    # ========================================================

    st.header("📅 Order Information")

    col1, col2 = st.columns(2)


    with col1:

        order_date = st.date_input(
            "Order Date",
            value=min_order_date,
            min_value=min_order_date,
            max_value=max_order_date,
            format="YYYY/MM/DD"
        )


    with col2:

        order_time = st.time_input(
            "Order Time"
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.divider()

    predict_button = st.form_submit_button(
        "🔮 Predict Late Delivery Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:


    # ========================================================
    # CREATE DATETIME
    # ========================================================

    order_datetime = pd.Timestamp(
        f"{order_date} {order_time}"
    )


    # ========================================================
    # DERIVE TEMPORAL FEATURES
    # ========================================================

    order_month = order_datetime.month

    order_day = order_datetime.day

    order_day_of_week = (
        order_datetime.dayofweek
    )

    order_hour = order_datetime.hour

    is_weekend = int(
        order_day_of_week >= 5
    )

    order_week = int(
        order_datetime.isocalendar().week
    )

    order_day_of_month = (
        order_datetime.day
    )

    is_month_start = int(
        order_datetime.is_month_start
    )

    is_month_end = int(
        order_datetime.is_month_end
    )

    is_quarter_start = int(
        order_datetime.is_quarter_start
    )

    is_quarter_end = int(
        order_datetime.is_quarter_end
    )

    is_business_hour = int(
        order_datetime.dayofweek < 5
        and 9 <= order_datetime.hour < 17
    )

    is_night = int(
        order_datetime.hour < 6
        or order_datetime.hour >= 22
    )


    # ========================================================
    # INTERACTION FEATURE
    # ========================================================

    interaction_shipping_hour = (
        shipping_mode
        + "_"
        + str(order_hour)
    )


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame([{

        "Shipping Mode":
            shipping_mode,

        "Days for shipment (scheduled)":
            scheduled_days,

        "Customer Country":
            customer_country,

        "Category Name":
            category_name,

        "Order Item Quantity":
            quantity,

        "Market":
            market,

        "Order Region":
            order_region,

        "Order Country":
            order_country,

        "Order State":
            order_state,

        "Customer Segment":
            customer_segment,

        "Department Name":
            department_name,

        "Order Item Product Price":
            product_price,

        "Order Item Discount Rate":
            discount_rate,

        "Sales":
            sales,

        "Order Month":
            order_month,

        "Order Day":
            order_day,

        "Order Day of Week":
            order_day_of_week,

        "Order Hour":
            order_hour,

        "Is Weekend":
            is_weekend,

        "Order Week of Year":
            order_week,

        "Order Day of Month":
            order_day_of_month,

        "Is Month Start":
            is_month_start,

        "Is Month End":
            is_month_end,

        "Is Quarter Start":
            is_quarter_start,

        "Is Quarter End":
            is_quarter_end,

        "Is Business Hour":
            is_business_hour,

        "Is Night":
            is_night,

        "Interaction_ShippingMode_OrderHour":
            interaction_shipping_hour,

        "Customer City":
            customer_city,

        "Type":
            shipment_type,

        "Customer State":
            customer_state

    }])


    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    try:

        probability = model.predict_proba(
            input_data
        )[0][1]

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        st.stop()


    # ========================================================
    # APPLY THRESHOLD
    # ========================================================

    prediction = int(
        probability >= threshold
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.header("📊 Prediction Result")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Late Delivery Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK OF LATE DELIVERY"
        )

        st.write(
            "The model predicts that this shipment "
            "has a high risk of late delivery."
        )

    else:

        st.success(
            "✅ LOW RISK / LIKELY ON TIME"
        )

        st.write(
            "The model predicts that this shipment "
            "has a low risk of late delivery."
        )