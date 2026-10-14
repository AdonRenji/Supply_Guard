
from pathlib import Path
import json
import os
from datetime import datetime
from functools import wraps

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash


# --------------------------------------------------
# 1. PROJECT PATHS AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"
DATASET_PATH = BASE_DIR / "DataCoSupplyChainDataset.csv"
USERS_PATH = BASE_DIR / "users.json"

MODEL_PATH = MODEL_DIR / "supplyguard_19_model.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"

app = Flask(__name__)

# Set SUPPLYGUARD_SECRET_KEY in your environment before deployment.
app.secret_key = os.environ.get(
    "SUPPLYGUARD_SECRET_KEY",
    "development-only-change-this-secret-key"
)

THRESHOLD = 0.41

FEATURES = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "Sales",
    "Type",
    "Order Month",
    "Order Day",
    "Order Day of Week",
    "Order Hour",
    "Is Weekend",
    "Order Week of Year",
    "Is Night",
    "Interaction_ShippingMode_OrderHour",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)


# --------------------------------------------------
# 2. USER STORAGE USING users.json
# --------------------------------------------------

def load_users():
    """Load registered accounts from the JSON file."""
    if not USERS_PATH.exists():
        USERS_PATH.write_text("{}", encoding="utf-8")

    try:
        with open(USERS_PATH, "r", encoding="utf-8") as file:
            users = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        app.logger.exception("Could not read users.json.")
        raise RuntimeError(
            "users.json is invalid or cannot be read. "
            "Check the file before continuing."
        ) from exc

    if not isinstance(users, dict):
        raise RuntimeError("users.json must contain a JSON object, such as {}.")

    return users


def save_users(users):
    """Save user accounts to users.json."""
    temporary_path = USERS_PATH.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=4, ensure_ascii=False)

    temporary_path.replace(USERS_PATH)


def login_required(view_function):
    """Require a logged-in user before opening a protected page."""
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "email" not in session:
            flash("Please log in to access SupplyGuard.", "error")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped_view


# --------------------------------------------------
# 3. REGISTRATION
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if "email" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password or not confirm_password:
            flash("Please complete all fields.", "error")

        elif len(name) > 100:
            flash("Your name must be 100 characters or fewer.", "error")

        elif len(email) > 254:
            flash("Please enter a valid email address.", "error")

        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")

        elif password != confirm_password:
            flash("Passwords do not match.", "error")

        else:
            try:
                users = load_users()

                if email in users:
                    flash("An account with this email already exists.", "error")
                else:
                    users[email] = {
                        "name": name,
                        "password_hash": generate_password_hash(password),
                    }

                    save_users(users)
                    flash("Account created successfully. Please log in.", "success")
                    return redirect(url_for("login"))

            except (OSError, RuntimeError):
                app.logger.exception("Registration could not be completed.")
                flash(
                    "Could not save your account. Check users.json permissions.",
                    "error",
                )

    return render_template("register.html")


# --------------------------------------------------
# 4. LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if "email" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        try:
            users = load_users()
            user = users.get(email)

            if (
                user
                and isinstance(user, dict)
                and user.get("password_hash")
                and check_password_hash(user["password_hash"], password)
            ):
                session.clear()
                session["email"] = email
                session["name"] = user.get("name", "User")

                flash("Login successful.", "success")
                return redirect(url_for("home"))

            flash("Invalid email or password.", "error")

        except (OSError, RuntimeError):
            app.logger.exception("Login could not be completed.")
            flash("Unable to access user accounts. Check users.json.", "error")

    return render_template("login.html")


# --------------------------------------------------
# 5. LOAD DROPDOWN OPTIONS FROM THE DATASET
# --------------------------------------------------

dataset = pd.read_csv(
    DATASET_PATH,
    encoding="cp1252",
    low_memory=False,
)

DATE_COLUMN = "order date (DateOrders)"

if DATE_COLUMN not in dataset.columns:
    raise ValueError(f"Missing dataset column: {DATE_COLUMN}")


def get_options(column):
    if column not in dataset.columns:
        raise ValueError(f"Missing dataset column: {column}")

    values = dataset[column].dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


options = {
    "shipping_mode": get_options("Shipping Mode"),
    "market": get_options("Market"),
    "order_region": get_options("Order Region"),
    "order_country": get_options("Order Country"),
    "order_state": get_options("Order State"),
    "type": get_options("Type"),
}

days_options = sorted(
    pd.to_numeric(
        dataset["Days for shipment (scheduled)"], errors="coerce"
    ).dropna().unique().tolist()
)

quantity_options = sorted(
    pd.to_numeric(
        dataset["Order Item Quantity"], errors="coerce"
    ).dropna().unique().tolist()
)

discount_values = [0.05, 0.10, 0.25, 0.50, 0.75]

discount_options = [
    {"value": f"{val:.2f}", "label": f"{int(round(val * 100))}% discount"}
    for val in discount_values
]

parsed_dates = pd.to_datetime(
    dataset[DATE_COLUMN], errors="coerce"
).dropna()

min_date = parsed_dates.min().strftime("%Y-%m-%d")
max_date = datetime.now().strftime("%Y-%m-%d")


# --------------------------------------------------
# 6. PREPARE THE EXACT 19 MODEL FEATURES
# --------------------------------------------------

def prepare_features(form_data):
    shipping_mode = form_data.get("shipping_mode", "").strip()
    market = form_data.get("market", "").strip()
    order_region = form_data.get("order_region", "").strip()
    order_country = form_data.get("order_country", "").strip()
    order_state = form_data.get("order_state", "").strip()
    order_type = form_data.get("type", "").strip()

    categories = [
        shipping_mode, market, order_region,
        order_country, order_state, order_type
    ]

    if not all(categories):
        raise ValueError("Please complete all categorical fields.")

    try:
        scheduled_days = float(form_data["days_scheduled"])
        quantity = float(form_data["quantity"])
        product_price = float(form_data["product_price"])
        discount_rate = float(form_data["discount_rate"])
        sales = float(form_data["sales"])

        order_date_text = form_data["order_date"].strip()
        order_time_text = form_data.get("order_time", "").strip()
        order_period = form_data.get("order_period", "").strip().upper()

        if not order_time_text and "order_time_manual" in form_data:
            order_time_text = form_data.get("order_time_manual", "").strip()

        if " " in order_time_text:
            parts = order_time_text.split()
            order_time_text = parts[0]
            if len(parts) > 1 and parts[1].upper() in ["AM", "PM"]:
                order_period = parts[1].upper()

        if ":" not in order_time_text:
            raise ValueError("Enter order time in HH:MM format (e.g. 09:30).")

        hour_text, minute_text = order_time_text.split(":", 1)
        try:
            hour = int(hour_text)
            minute = int(minute_text)
        except ValueError as exc:
            raise ValueError("Enter valid numbers for hour and minute.") from exc

        if not (0 <= minute <= 59):
            raise ValueError("Minute must be between 00 and 59.")

        if order_period in ["AM", "PM"]:
            if not (1 <= hour <= 12):
                if not (0 <= hour <= 23):
                    raise ValueError("Hour must be between 1 and 12 when selecting AM/PM.")
            else:
                if order_period == "PM" and hour != 12:
                    hour += 12
                elif order_period == "AM" and hour == 12:
                    hour = 0
        else:
            if not (0 <= hour <= 23):
                raise ValueError("Hour must be between 00 and 23.")

        order_time_text = f"{hour:02d}:{minute:02d}"

        combined_datetime = f"{order_date_text} {order_time_text}:00"

        order_datetime = pd.to_datetime(
            combined_datetime,
            format="%Y-%m-%d %H:%M:%S",
            errors="raise",
        )

    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(
            "Enter valid shipment details, date, time and numeric values."
        ) from exc

    numeric_values = [
        scheduled_days, quantity, product_price,
        discount_rate, sales
    ]

    if not all(pd.notna(value) for value in numeric_values):
        raise ValueError("Numeric fields cannot be empty.")

    if any(value < 0 for value in numeric_values):
        raise ValueError("Numeric values cannot be negative.")

    if scheduled_days not in days_options:
        raise ValueError("Select a scheduled shipping day value from the list.")

    if quantity not in quantity_options or quantity < 1:
        raise ValueError("Select a valid order quantity.")

    if not (0.0 <= discount_rate <= 1.0):
        raise ValueError("Select a valid discount rate between 0 and 1.")

    current_max_date = datetime.now().strftime("%Y-%m-%d")
    if not (min_date <= order_date_text <= current_max_date):
        raise ValueError(
            f"Choose an order date between {min_date} and {current_max_date}."
        )

    day_of_week = int(order_datetime.dayofweek)
    order_hour = int(order_datetime.hour)

    row = {
        "Shipping Mode": shipping_mode,
        "Days for shipment (scheduled)": scheduled_days,
        "Order Item Quantity": quantity,
        "Market": market,
        "Order Region": order_region,
        "Order Country": order_country,
        "Order State": order_state,
        "Order Item Product Price": product_price,
        "Order Item Discount Rate": discount_rate,
        "Sales": sales,
        "Type": order_type,
        "Order Month": int(order_datetime.month),
        "Order Day": int(order_datetime.day),
        "Order Day of Week": day_of_week,
        "Order Hour": order_hour,
        "Is Weekend": int(day_of_week in [5, 6]),
        "Order Week of Year": float(order_datetime.isocalendar().week),
        "Is Night": int(order_hour < 6 or order_hour >= 22),
        "Interaction_ShippingMode_OrderHour": f"{shipping_mode}_{order_hour}",
    }

    return pd.DataFrame([row], columns=FEATURES)


# --------------------------------------------------
# 7. HOME PAGE AND SHIPMENT PREDICTION
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    result = None
    error = None
    form_data = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            shipment_features = prepare_features(request.form)
            probabilities = model.predict_proba(shipment_features)

            classes = list(model.classes_)
            if 1 not in classes:
                raise RuntimeError("The model does not contain class 1.")

            late_probability = float(
                probabilities[0, classes.index(1)]
            )

            is_late = late_probability >= THRESHOLD

            result = {
                "prediction": int(is_late),
                "probability": late_probability * 100,
                "label": (
                    "Late-Delivery Risk Detected"
                    if is_late
                    else "Lower Predicted Late-Delivery Risk"
                ),
            }

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Shipment prediction failed.")
            error = (
                "Prediction failed. Check the inputs and confirm that "
                "the saved model matches train_model.py."
            )

    return render_template(
        "home.html",
        options=options,
        days_options=days_options,
        quantity_options=quantity_options,
        discount_options=discount_options,
        min_date=min_date,
        max_date=max_date,
        form_data=form_data,
        result=result,
        error=error,
    )


# --------------------------------------------------
# 8. LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# --------------------------------------------------
# 9. RUN LOCALLY
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
