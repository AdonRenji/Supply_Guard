
from pathlib import Path
import json
import os
import random
import sqlite3
from datetime import datetime
from functools import wraps

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

import database


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

# Initialize SQLite database and seed products
database.init_db(DATASET_PATH)


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

    try:
        temporary_path.replace(USERS_PATH)
    except PermissionError:
        import time
        time.sleep(0.15)
        try:
            temporary_path.replace(USERS_PATH)
        except PermissionError:
            with open(USERS_PATH, "w", encoding="utf-8") as file:
                json.dump(users, file, indent=4, ensure_ascii=False)
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)


def login_required(view_function):
    """Require a logged-in user before opening a protected page."""
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "email" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped_view


def manager_required(view_function):
    """Require a logged-in manager before opening manager dashboard or risk actions."""
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "email" not in session:
            return redirect(url_for("login"))

        if session.get("role") != "manager":
            flash("Access denied. Manager operations require a manager account.", "error")
            return redirect(url_for("place_order"))

        return view_function(*args, **kwargs)

    return wrapped_view


def customer_required(view_function):
    """Require a logged-in user to place or view customer orders."""
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "email" not in session:
            flash("Please log in or register to place your order.", "error")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped_view


# --------------------------------------------------
# 3. REGISTRATION
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    target_role = request.args.get("role", "customer").strip().lower()
    if target_role not in ["customer", "manager"]:
        target_role = "customer"

    if "email" in session:
        if session.get("role") == "customer":
            return redirect(url_for("place_order"))
        elif session.get("role") == "manager" and target_role != "manager":
            return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", target_role).strip().lower()

        if role not in ["customer", "manager"]:
            role = target_role

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
                        "role": role,
                        "password_hash": generate_password_hash(password),
                    }

                    save_users(users)
                    if "email" in session and session.get("role") == "manager":
                        flash(f"New Manager account for '{name}' ({email}) created successfully!", "success")
                        return redirect(url_for("home"))

                    portal_name = "Customer Portal" if role == "customer" else "Manager Operations"
                    flash(f"Account created successfully as {role.capitalize()}! Please sign in via the {portal_name}.", "success")
                    return redirect(url_for("login", portal=role))

            except (OSError, RuntimeError):
                app.logger.exception("Registration could not be completed.")
                flash(
                    "Could not save your account. Check users.json permissions.",
                    "error",
                )

    return render_template("register.html", role=target_role)


# --------------------------------------------------
# 4. LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if "email" in session:
        if session.get("role") == "customer":
            return redirect(url_for("place_order"))
        return redirect(url_for("home"))

    active_portal = request.args.get("portal", "customer").strip().lower()
    if active_portal not in ["customer", "manager"]:
        active_portal = "customer"

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        login_portal = request.form.get("login_portal", active_portal).strip().lower()
        if login_portal not in ["customer", "manager"]:
            login_portal = "customer"

        try:
            users = load_users()
            user = users.get(email)

            if (
                user
                and isinstance(user, dict)
                and user.get("password_hash")
                and check_password_hash(user["password_hash"], password)
            ):
                actual_role = user.get("role", "customer").lower()

                # Strict Portal Scope Enforcement
                if login_portal == "customer" and actual_role != "customer":
                    flash("ℹ️ This is a Manager account. Please select the 'Manager Operations' tab above to sign in.", "error")
                    return render_template("login.html", active_portal="manager")

                if login_portal == "manager" and actual_role != "manager":
                    flash("⚠️ Access Denied: This is a Customer account. Please select the 'Customer Portal' tab above to sign in.", "error")
                    return render_template("login.html", active_portal="customer")

                session.clear()
                session["email"] = email
                session["name"] = user.get("name", "User")
                session["role"] = actual_role

                flash(f"Welcome back, {session['name']}!", "success")
                if actual_role == "customer":
                    return redirect(url_for("place_order"))
                else:
                    return redirect(url_for("home"))

            flash("Invalid email or password.", "error")
            return render_template("login.html", active_portal=login_portal)

        except (OSError, RuntimeError):
            app.logger.exception("Login could not be completed.")
            flash("Unable to access user accounts. Check users.json.", "error")

    return render_template("login.html", active_portal=active_portal)


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


def load_or_build_geo_hierarchy():
    """
    Load precomputed geographic hierarchy (Market -> Region -> Country -> States)
    or build and cache it from the dataset for fast cascading dropdowns.
    """
    geo_cache_path = MODEL_DIR / "geo_hierarchy.json"

    if geo_cache_path.exists():
        try:
            with open(geo_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            app.logger.warning("Could not read cached geo_hierarchy.json; regenerating.")

    geo_cols = ["Market", "Order Region", "Order Country", "Order State"]
    geo_df = dataset[geo_cols].dropna().drop_duplicates().sort_values(geo_cols)

    hierarchy = {}
    for row in geo_df.itertuples(index=False):
        market = str(row[0]).strip()
        region = str(row[1]).strip()
        country = str(row[2]).strip()
        state = str(row[3]).strip()

        if not (market and region and country and state):
            continue

        if market not in hierarchy:
            hierarchy[market] = {}
        if region not in hierarchy[market]:
            hierarchy[market][region] = {}
        if country not in hierarchy[market][region]:
            hierarchy[market][region][country] = []

        if state not in hierarchy[market][region][country]:
            hierarchy[market][region][country].append(state)

    try:
        with open(geo_cache_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    except Exception:
        app.logger.warning("Could not save geo_hierarchy.json cache.")

    return hierarchy


geo_hierarchy = load_or_build_geo_hierarchy()

GEO_MAPPING_PATH = MODEL_DIR / "geo_mapping.json"
geo_mapping = {}
if GEO_MAPPING_PATH.exists():
    try:
        with open(GEO_MAPPING_PATH, "r", encoding="utf-8") as f:
            geo_mapping = json.load(f)
    except Exception:
        app.logger.warning("Could not read geo_mapping.json")

def to_model_geo(name):
    """Translate clean user-facing English name back to the dataset's native string for ML prediction."""
    return geo_mapping.get(name, name)

options = {
    "shipping_mode": get_options("Shipping Mode"),
    "market": sorted(list(geo_hierarchy.keys())),
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

    if market not in geo_hierarchy:
        raise ValueError("Please select a valid market.")

    if order_region not in geo_hierarchy.get(market, {}):
        raise ValueError(f"Selected region '{order_region}' does not belong to market '{market}'.")

    if order_country not in geo_hierarchy.get(market, {}).get(order_region, {}):
        raise ValueError(f"Selected country '{order_country}' does not belong to region '{order_region}'.")

    if order_state not in geo_hierarchy.get(market, {}).get(order_region, {}).get(order_country, []):
        raise ValueError(f"Selected state '{order_state}' does not belong to country '{order_country}'.")

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

    shipping_mode_days_map = {
        "Same Day": 0,
        "First Class": 1,
        "Second Class": 2,
        "Standard Class": 4,
    }
    expected_days = shipping_mode_days_map.get(shipping_mode)
    if expected_days is not None and int(scheduled_days) != expected_days:
        raise ValueError(
            f"Shipping mode '{shipping_mode}' corresponds to {expected_days} scheduled shipping day(s)."
        )

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
        "Market": to_model_geo(market),
        "Order Region": to_model_geo(order_region),
        "Order Country": to_model_geo(order_country),
        "Order State": to_model_geo(order_state),
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

def build_prediction_result(row_dict, late_probability, pred_id=None, order_id=None, order=None):
    if pred_id is None:
        pred_id = f"SG-{datetime.now().strftime('%Y%m%d')}-{random.randint(100, 999)}"
    pred_timestamp = datetime.now().strftime("%b %d, %Y | %H:%M")
    is_late = late_probability >= THRESHOLD

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    day_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    month_num = int(row_dict["Order Month"])
    month_name = month_names[month_num - 1] if 1 <= month_num <= 12 else str(month_num)

    day_num = int(row_dict["Order Day of Week"])
    day_name = day_names[day_num] if 0 <= day_num <= 6 else str(day_num)

    is_weekend_bool = bool(row_dict["Is Weekend"])
    is_night_bool = bool(row_dict["Is Night"])

    region_name = order.get("order_region") if order else row_dict.get("Order Region", "")

    mode = str(row_dict.get("Shipping Mode", ""))
    sched_days = int(row_dict.get("Days for shipment (scheduled)", 0))

    time_of_day = "night" if is_night_bool else "daytime"
    day_type = "weekend" if is_weekend_bool else "weekday"

    order_hour_val = int(row_dict.get("Order Hour", 12))
    country_name = order.get("order_country") if order else str(row_dict.get("Order Country", ""))
    order_type = str(row_dict.get("Type", "DEBIT"))

    total_qty = 1
    if order and order.get("items"):
        total_qty = sum(it.get("quantity", 1) for it in order["items"])
    else:
        total_qty = int(row_dict.get("Order Item Quantity", 1))

    dest_str = country_name if country_name else region_name

    if is_late:
        # Simple timing reason
        if is_night_bool:
            timing_phrase = f"placed late at night ({order_hour_val:02d}:00) after carrier cut-off"
        elif 18 <= order_hour_val <= 21:
            timing_phrase = f"placed in the evening ({order_hour_val:02d}:00) after daily dispatch"
        elif is_weekend_bool:
            timing_phrase = f"placed on a weekend ({day_name}) during reduced carrier operations"
        else:
            timing_phrase = f"placed on {day_name} at {order_hour_val:02d}:00"

        # Simple mode & schedule reason
        if mode == "Same Day" or sched_days == 0:
            mode_phrase = "demands same-day delivery with no extra margin for delays"
        elif mode == "First Class" or sched_days == 1:
            mode_phrase = f"has a tight 1-day delivery schedule to {dest_str}"
        elif mode == "Second Class" or sched_days == 2:
            mode_phrase = f"allows only 2 days for transit to {dest_str}"
        else:
            mode_phrase = f"uses Standard Class ({sched_days} scheduled days), which frequently faces hub sorting delays"

        # Simple 1-sentence key insight
        key_insight = f"High delay risk was predicted because this order was {timing_phrase}, and {mode_phrase}."

        # Standard actionable recommendations for all late-delivery orders
        destination_label = dest_str if dest_str else "the destination"
        rec_1 = "Prioritize warehouse packing immediately for the next courier pickup."
        rec_2 = f"Dispatch via a direct local courier to {destination_label}."
        rec_3 = "Share live delivery tracking with the customer."

        recommendations = [rec_1, rec_2, rec_3]

    else:
        # Simple timing for on-time
        timing_phrase = f"placed during normal business hours ({order_hour_val:02d}:00 on {day_name})"

        # Simple mode & schedule for on-time
        if mode == "Standard Class":
            mode_phrase = f"the {sched_days}-day scheduled window gives enough time for standard delivery to {dest_str}"
        elif mode in ["First Class", "Second Class"]:
            mode_phrase = f"{mode} priority shipping provides fast and reliable transit to {dest_str}"
        else:
            mode_phrase = "early dispatch allows same-day delivery during regular daytime hours"

        key_insight = f"Low delay risk was predicted because this order was {timing_phrase}, and {mode_phrase}."

        rec_1 = f"Pack this order following standard {mode} procedures."
        rec_2 = f"Hand over the parcel during regular carrier pickup on {day_name}."
        rec_3 = f"Provide tracking details confirming delivery to {dest_str} within {sched_days} business days."

        recommendations = [rec_1, rec_2, rec_3]

    # Dynamic Contributing Factors for Machine Learning Prediction Explanation UI
    dest_display = f"{region_name} ({dest_str})" if dest_str and dest_str != region_name else (region_name or dest_str or "Global")

    if is_late:
        # 1. Shipping Mode
        if mode == "Standard Class":
            mode_score = 82
            mode_impact_level = "high"
            mode_impact_label = "High Delay Impact"
            mode_bar_class = "bar-high"
            mode_impact_class = "impact-danger"
            mode_exp = "Standard Class passes through multiple regional sorting hubs, which frequently causes transit delays."
        elif mode == "Same Day":
            mode_score = 88
            mode_impact_level = "critical"
            mode_impact_label = "Zero-Margin Schedule"
            mode_bar_class = "bar-critical"
            mode_impact_class = "impact-critical"
            mode_exp = "Same Day shipping requires immediate fulfillment, leaving no buffer for warehouse processing."
        elif mode == "First Class":
            mode_score = 74
            mode_impact_level = "high"
            mode_impact_label = "Tight Turnaround"
            mode_bar_class = "bar-high"
            mode_impact_class = "impact-danger"
            mode_exp = "First Class requires fast 1-day delivery, so any small sorting delay causes a late arrival."
        else:
            mode_score = 64
            mode_impact_level = "moderate"
            mode_impact_label = "Limited Buffer"
            mode_bar_class = "bar-medium"
            mode_impact_class = "impact-warning"
            mode_exp = "Second Class allows only 2 days, leaving little room for transit hold-ups."

        # 2. Transit Time & Scheduling
        if sched_days <= 1:
            transit_score = 85
            transit_impact_level = "critical"
            transit_impact_label = "Tight Delivery Window"
            transit_bar_class = "bar-critical"
            transit_impact_class = "impact-critical"
            if sched_days == 0:
                transit_exp = "A 0-day scheduled window requires delivery today, leaving zero buffer for carrier transit."
            else:
                transit_exp = "A 1-day scheduled window requires next-day delivery, leaving minimal buffer for carrier transit."
        elif sched_days >= 4:
            transit_score = 76
            transit_impact_level = "high"
            transit_impact_label = "Long Transit Route"
            transit_bar_class = "bar-high"
            transit_impact_class = "impact-danger"
            transit_exp = f"{sched_days} scheduled transit days involve multiple hub transfers, increasing delay risk."
        else:
            transit_score = 62
            transit_impact_level = "moderate"
            transit_impact_label = "Short Buffer Margin"
            transit_bar_class = "bar-medium"
            transit_impact_class = "impact-warning"
            transit_exp = f"{sched_days} scheduled transit days provide less than 24 hours of buffer for delays."

        # 3. Destination Region
        dest_score = 69
        dest_impact_level = "high"
        dest_impact_label = "Regional Route Risk"
        dest_bar_class = "bar-high"
        dest_impact_class = "impact-danger"
        dest_exp = f"Shipments to {dest_display} pass through multiple regional hubs, increasing delay risk."

        # 4. Dispatch Timing
        if is_night_bool:
            time_score = 89
            time_impact_level = "critical"
            time_impact_label = "Late-Night Placement"
            time_bar_class = "bar-critical"
            time_impact_class = "impact-critical"
            time_exp = f"Order was placed at {order_hour_val:02d}:00 after carrier cut-off, delaying dispatch to the next morning."
        elif is_weekend_bool:
            time_score = 80
            time_impact_level = "high"
            time_impact_label = "Weekend Placement"
            time_bar_class = "bar-high"
            time_impact_class = "impact-danger"
            time_exp = f"Order was placed on {day_name} when carriers operate on reduced sorting schedules."
        elif order_hour_val >= 16:
            time_score = 72
            time_impact_level = "moderate"
            time_impact_label = "Near Cut-Off Time"
            time_bar_class = "bar-medium"
            time_impact_class = "impact-warning"
            time_exp = f"Order booked at {order_hour_val:02d}:00 is close to the daily carrier cut-off."
        else:
            time_score = 56
            time_impact_level = "moderate"
            time_impact_label = "Peak Queue Hours"
            time_bar_class = "bar-medium"
            time_impact_class = "impact-warning"
            time_exp = f"Order placed at {order_hour_val:02d}:00 during peak warehouse fulfillment hours."

        # Drivers
        if is_night_bool:
            primary_driver = f"Late-Night Order Placement ({order_hour_val:02d}:00 after carrier cut-off)"
            secondary_driver = f"Shipping Mode ({mode} - {sched_days} scheduled days)"
        elif mode in ["Same Day", "First Class"]:
            primary_driver = f"Tight Delivery Window ({sched_days} scheduled days)"
            secondary_driver = f"Destination Route ({region_name})"
        elif is_weekend_bool:
            primary_driver = f"Weekend Timing ({day_name} reduced carrier shifts)"
            secondary_driver = f"Shipping Mode ({mode})"
        else:
            primary_driver = f"Standard Class Transit Hub Stops"
            secondary_driver = f"Destination Region ({region_name})"

    else:
        # On-Time / Low Risk
        if mode in ["First Class", "Second Class"]:
            mode_score = 22
            mode_impact_level = "optimal"
            mode_impact_label = "Priority Shipping"
            mode_bar_class = "bar-low"
            mode_impact_class = "impact-success"
            mode_exp = f"{mode} moves quickly through carrier hubs with minimal waiting time."
        elif mode == "Standard Class":
            mode_score = 30
            mode_impact_level = "favorable"
            mode_impact_label = "Adequate Transit Time"
            mode_bar_class = "bar-low"
            mode_impact_class = "impact-success"
            mode_exp = "Standard Class scheduled days give enough time for normal delivery."
        else:
            mode_score = 25
            mode_impact_level = "optimal"
            mode_impact_label = "Same-Day Route"
            mode_bar_class = "bar-low"
            mode_impact_class = "impact-success"
            mode_exp = "Local courier can complete same-day delivery during regular hours."

        transit_score = 18
        transit_impact_level = "optimal"
        transit_impact_label = "Enough Buffer Time"
        transit_bar_class = "bar-low"
        transit_impact_class = "impact-success"
        transit_exp = f"The {sched_days}-day scheduled window provides plenty of buffer for any minor delays."

        dest_score = 24
        dest_impact_level = "optimal"
        dest_impact_label = "Direct Carrier Route"
        dest_bar_class = "bar-low"
        dest_impact_class = "impact-success"
        dest_exp = f"Carrier routes to {dest_display} are direct and reliable."

        time_score = 15
        time_impact_level = "optimal"
        time_impact_label = "Optimal Order Time"
        time_bar_class = "bar-low"
        time_impact_class = "impact-success"
        time_exp = f"Order was placed during business hours ({order_hour_val:02d}:00 on {day_name}) with ample packing time."

        primary_driver = f"Sufficient Transit Buffer ({sched_days} scheduled days)"
        secondary_driver = f"Normal Business Hours ({order_hour_val:02d}:00 on {day_name})"

    contributing_factors = {
        "shipping_mode": {
            "score": mode_score,
            "impact_level": mode_impact_level,
            "impact_label": mode_impact_label,
            "impact_class": mode_impact_class,
            "bar_class": mode_bar_class,
            "explanation": mode_exp,
        },
        "transit_time": {
            "score": transit_score,
            "impact_level": transit_impact_level,
            "impact_label": transit_impact_label,
            "impact_class": transit_impact_class,
            "bar_class": transit_bar_class,
            "explanation": transit_exp,
        },
        "destination_region": {
            "score": dest_score,
            "impact_level": dest_impact_level,
            "impact_label": dest_impact_label,
            "impact_class": dest_impact_class,
            "bar_class": dest_bar_class,
            "explanation": dest_exp,
            "route_display": dest_display,
        },
        "dispatch_timing": {
            "score": time_score,
            "impact_level": time_impact_level,
            "impact_label": time_impact_label,
            "impact_class": time_impact_class,
            "bar_class": time_bar_class,
            "explanation": time_exp,
        },
        "primary_driver": primary_driver,
        "secondary_driver": secondary_driver,
    }

    # Circle circumference for r=40 is ~251.2
    stroke_offset = round(251.2 - (251.2 * late_probability), 1)

    has_multiple_items = order and len(order.get("items", [])) > 1

    if has_multiple_items:
        qty_display = f"{sum(it['quantity'] for it in order['items'])} items"
        price_display = "Itemized in order"
        disc_display = "Itemized in order"
        sales_display = f"${float(order.get('total_amount', row_dict.get('Sales', 0.0))):.2f}"
    else:
        qty_display = int(row_dict["Order Item Quantity"])
        price_display = f"${float(row_dict['Order Item Product Price']):.2f}"
        disc_display = f"{int(round(float(row_dict['Order Item Discount Rate']) * 100))}%"
        sales_display = f"${float(row_dict['Sales']):.2f}"

    result_data = {
        "prediction_id": pred_id,
        "order_id": order_id,
        "timestamp": pred_timestamp,
        "is_late": is_late,
        "label": "High Risk" if is_late else "Low Risk",
        "description": (
            "This shipment has a high chance of being delayed."
            if is_late
            else "This shipment is expected to arrive on schedule."
        ),
        "probability": late_probability * 100,
        "probability_int": int(round(late_probability * 100)),
        "stroke_offset": stroke_offset,
        "shipping_mode": row_dict["Shipping Mode"],
        "days_scheduled": f"{int(row_dict['Days for shipment (scheduled)'])} days",
        "interaction_mode_hour": row_dict.get("Interaction_ShippingMode_OrderHour", f"{row_dict.get('Shipping Mode')}_{int(row_dict.get('Order Hour', 0))}"),
        "market": order.get("market") if order else row_dict.get("Market"),
        "order_region": order.get("order_region") if order else row_dict.get("Order Region"),
        "order_country": order.get("order_country") if order else row_dict.get("Order Country"),
        "order_state": order.get("order_state") if order else row_dict.get("Order State"),
        "quantity": qty_display,
        "product_price": price_display,
        "discount_rate": disc_display,
        "sales": sales_display,
        "order_type": row_dict["Type"],
        "order_month": month_name,
        "order_day": int(row_dict["Order Day"]),
        "order_day_name": day_name,
        "order_hour": f"{int(row_dict['Order Hour']):02d}",
        "is_weekend": "Yes" if is_weekend_bool else "No",
        "order_week": int(row_dict["Order Week of Year"]),
        "is_night": "Yes" if is_night_bool else "No",
        "recommendations": recommendations,
        "key_insight": key_insight,
        "contributing_factors": contributing_factors,
    }
    return result_data


# --------------------------------------------------
# 7. HOME PAGE AND SHIPMENT PREDICTION
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@manager_required
def home():
    lookup_id = request.args.get("lookup_id") or request.form.get("lookup_id")
    retrieved_order = None
    lookup_error = None

    if lookup_id:
        lookup_id = lookup_id.strip()
        retrieved_order = database.get_order_by_order_id(lookup_id)
        if not retrieved_order:
            lookup_error = f"No order found matching Order ID '{lookup_id}'. Please verify the ID."

    recent_orders = database.get_recent_orders(limit=10)

    # Deduplicate flash messages in session so they never appear multiple times
    if "_flashes" in session and session["_flashes"]:
        unique_flashes = []
        for cat, msg in session["_flashes"]:
            if (cat, msg) not in unique_flashes:
                unique_flashes.append((cat, msg))
        session["_flashes"] = unique_flashes

    return render_template(
        "home.html",
        retrieved_order=retrieved_order,
        lookup_error=lookup_error,
        lookup_id=lookup_id or "",
        recent_orders=recent_orders,
    )


@app.route("/manual-assessment", methods=["GET", "POST"])
@manager_required
def manual_assessment():
    error = None
    form_data = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            shipment_features = prepare_features(request.form)
            probabilities = model.predict_proba(shipment_features)

            classes = list(model.classes_)
            if 1 not in classes:
                raise RuntimeError("The model does not contain class 1.")

            late_probability = float(probabilities[0, classes.index(1)])
            row_dict = shipment_features.iloc[0].to_dict()
            result_data = build_prediction_result(row_dict, late_probability)
            return render_template("result.html", result=result_data, order=None)

        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Shipment prediction failed.")
            error = "Prediction failed. Check the inputs and confirm that the saved model matches train_model.py."

    return render_template(
        "manual_assessment.html",
        options=options,
        days_options=days_options,
        quantity_options=quantity_options,
        discount_options=discount_options,
        min_date=min_date,
        max_date=max_date,
        form_data=form_data,
        geo_hierarchy=geo_hierarchy,
        error=error,
    )


# --------------------------------------------------
# 8. CUSTOMER ORDER PLACEMENT & CONFIRMATION
# --------------------------------------------------

@app.route("/order", methods=["GET", "POST"])
@customer_required
def place_order():
    error = None
    form_data = request.form.to_dict() if request.method == "POST" else {}
    products = database.get_products_catalogue()

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_hour_24 = now.hour
    current_minute = now.minute
    current_period = "PM" if current_hour_24 >= 12 else "AM"
    display_hour = current_hour_24 % 12
    if display_hour == 0:
        display_hour = 12
    current_time_display = f"{display_hour:02d}:{current_minute:02d}"
    current_time_24 = f"{current_hour_24:02d}:{current_minute:02d}"

    if request.method == "POST":
        try:
            now_submit = datetime.now()
            today_date = now_submit.strftime("%Y-%m-%d")
            submit_time_24 = now_submit.strftime("%H:%M")

            customer_name = request.form.get("customer_name", "").strip()
            customer_email = request.form.get("customer_email", "").strip().lower()
            order_type = request.form.get("order_type", "").strip()
            order_date = request.form.get("order_date", "").strip() or today_date
            order_time = request.form.get("order_time", "").strip()
            order_time_manual = request.form.get("order_time_manual", "").strip()
            order_period = request.form.get("order_period", "AM").strip().upper()

            shipping_mode = request.form.get("shipping_mode", "").strip()
            days_scheduled = request.form.get("days_scheduled", "").strip()
            market = request.form.get("market", "").strip()
            order_region = request.form.get("order_region", "").strip()
            order_country = request.form.get("order_country", "").strip()
            order_state = request.form.get("order_state", "").strip()

            if not customer_name:
                raise ValueError("Customer name is required.")
            if not customer_email or "@" not in customer_email:
                raise ValueError("Valid customer email is required.")
            if not order_type:
                raise ValueError("Payment / transaction type is required.")

            if not order_time and order_time_manual:
                if ":" in order_time_manual:
                    h_str, m_str = order_time_manual.split(":", 1)
                    try:
                        h = int(h_str)
                        m = int(m_str)
                        if order_period == "PM" and h != 12:
                            h += 12
                        elif order_period == "AM" and h == 12:
                            h = 0
                        order_time = f"{h:02d}:{m:02d}"
                    except ValueError:
                        order_time = submit_time_24
                else:
                    order_time = submit_time_24

            if not order_time:
                order_time = submit_time_24

            if not shipping_mode or not days_scheduled:
                raise ValueError("Shipping mode and scheduled days are required.")

            shipping_mode_days_map = {
                "Same Day": 0,
                "First Class": 1,
                "Second Class": 2,
                "Standard Class": 4,
            }
            exp_days = shipping_mode_days_map.get(shipping_mode)
            if exp_days is not None and int(float(days_scheduled)) != exp_days:
                raise ValueError(f"Shipping mode '{shipping_mode}' requires {exp_days} scheduled days.")

            if not (market and order_region and order_country and order_state):
                raise ValueError("Complete delivery destination is required.")

            if market not in geo_hierarchy:
                raise ValueError("Invalid market.")
            if order_region not in geo_hierarchy.get(market, {}):
                raise ValueError(f"Region '{order_region}' does not belong to market '{market}'.")
            if order_country not in geo_hierarchy.get(market, {}).get(order_region, {}):
                raise ValueError(f"Country '{order_country}' does not belong to region '{order_region}'.")
            if order_state not in geo_hierarchy.get(market, {}).get(order_region, {}).get(order_country, []):
                raise ValueError(f"State '{order_state}' does not belong to country '{order_country}'.")

            promo_code = request.form.get("promo_code", "").strip().upper()
            PROMO_MAP = {
                "": 0.0,
                "WELCOME10": 0.10,
                "SAVE15": 0.15,
                "SUPER20": 0.20,
                "SUPPLYGUARD": 0.15,
            }
            applied_discount = PROMO_MAP.get(promo_code, 0.0)

            product_ids = request.form.getlist("product_id[]")
            quantities = request.form.getlist("quantity[]")

            if not product_ids:
                raise ValueError("Order must contain at least one product.")

            items_list = []
            total_order_amount = 0.0

            for i in range(len(product_ids)):
                pid_str = product_ids[i].strip()
                if not pid_str:
                    continue
                pid = int(pid_str)
                prod = database.get_product_by_id(pid)
                if not prod:
                    raise ValueError(f"Selected product ID {pid} is not in the catalogue.")

                qty = int(quantities[i])
                if qty < 1 or qty > 50:
                    raise ValueError(f"Quantity for product {prod['product_name']} must be between 1 and 50.")

                disc = applied_discount
                item_sales = round(qty * prod["price"] * (1.0 - disc), 2)
                total_order_amount += item_sales

                items_list.append({
                    "product_id": pid,
                    "product_name": prod["product_name"],
                    "quantity": qty,
                    "product_price": prod["price"],
                    "discount_rate": disc,
                    "sales": item_sales,
                })

            if not items_list:
                raise ValueError("Please select at least one valid product.")

            order_data = {
                "customer_name": customer_name,
                "customer_email": customer_email,
                "shipping_mode": shipping_mode,
                "days_scheduled": float(days_scheduled),
                "market": market,
                "order_region": order_region,
                "order_country": order_country,
                "order_state": order_state,
                "order_type": order_type,
                "order_date": order_date,
                "order_time": order_time,
                "total_amount": round(total_order_amount, 2),
                "status": "CONFIRMED",
            }

            order_id, internal_id = database.save_customer_order(order_data, items_list)
            flash(f"Order {order_id} created successfully!", "success")
            return redirect(url_for("order_confirmation", order_id=order_id))

        except ValueError as exc:
            error = str(exc)
        except sqlite3.OperationalError as exc:
            app.logger.warning("Database lock detected: %s", exc)
            if "database is locked" in str(exc).lower():
                error = "Database is currently locked by another program (DB Browser for SQLite). Please close DB Browser (or click 'Write Changes' / 'Close Database') and try again."
            else:
                error = f"Database error: {exc}"
        except Exception:
            app.logger.exception("Failed to place order.")
            error = "Could not place order. Please review your inputs and try again."

    return render_template(
        "order.html",
        products=products,
        options=options,
        days_options=days_options,
        quantity_options=quantity_options,
        discount_options=discount_options,
        min_date=min_date,
        max_date=max_date,
        geo_hierarchy=geo_hierarchy,
        current_date=current_date,
        current_time_display=current_time_display,
        current_time_24=current_time_24,
        current_period=current_period,
        form_data=form_data,
        error=error,
    )


@app.route("/order-confirmation/<order_id>")
def order_confirmation(order_id):
    order = database.get_order_by_order_id(order_id)
    if not order:
        flash(f"Order ID '{order_id}' was not found.", "error")
        return redirect(url_for("place_order"))
    return render_template("order_confirmation.html", order=order)


# --------------------------------------------------
# 9. MANAGER ORDER LOOKUP & RISK PREDICTION
# --------------------------------------------------

@app.route("/order-lookup", methods=["GET", "POST"])
@manager_required
def order_lookup():
    order_id = request.values.get("order_id", "").strip()
    if order_id:
        return redirect(url_for("home", lookup_id=order_id))
    return redirect(url_for("home"))


@app.route("/predict-order/<order_id>", methods=["POST"])
@manager_required
def predict_order(order_id):
    order = database.get_order_by_order_id(order_id)
    if not order:
        flash(f"Order '{order_id}' not found.", "error")
        return redirect(url_for("home"))

    items = order.get("items", [])
    if not items:
        flash("Order has no associated items to evaluate.", "error")
        return redirect(url_for("home", lookup_id=order_id))

    try:
        order_datetime = pd.to_datetime(f"{order['order_date']} {order['order_time']}:00")
        day_of_week = int(order_datetime.dayofweek)
        order_hour = int(order_datetime.hour)

        item_rows = []
        for item in items:
            row = {
                "Shipping Mode": order["shipping_mode"],
                "Days for shipment (scheduled)": float(order["days_scheduled"]),
                "Order Item Quantity": float(item["quantity"]),
                "Market": to_model_geo(order["market"]),
                "Order Region": to_model_geo(order["order_region"]),
                "Order Country": to_model_geo(order["order_country"]),
                "Order State": to_model_geo(order["order_state"]),
                "Order Item Product Price": float(item["product_price"]),
                "Order Item Discount Rate": float(item["discount_rate"]),
                "Sales": float(item["sales"]),
                "Type": order["order_type"],
                "Order Month": int(order_datetime.month),
                "Order Day": int(order_datetime.day),
                "Order Day of Week": day_of_week,
                "Order Hour": order_hour,
                "Is Weekend": int(day_of_week in [5, 6]),
                "Order Week of Year": float(order_datetime.isocalendar().week),
                "Is Night": int(order_hour < 6 or order_hour >= 22),
                "Interaction_ShippingMode_OrderHour": f"{order['shipping_mode']}_{order_hour}",
            }
            item_rows.append(row)

        df_items = pd.DataFrame(item_rows, columns=FEATURES)
        probs = model.predict_proba(df_items)
        classes = list(model.classes_)
        class_1_idx = classes.index(1)

        item_probs = [float(probs[i, class_1_idx]) for i in range(len(items))]

        # In multi-product packages dispatched together, overall shipment delay risk
        # is determined by maximum risk among items.
        max_idx = item_probs.index(max(item_probs))
        overall_late_probability = item_probs[max_idx]
        overall_is_late = overall_late_probability >= THRESHOLD

        pred_id = f"SG-PRED-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        risk_label = "High Risk" if overall_is_late else "Low Risk"

        result_data = build_prediction_result(
            item_rows[max_idx],
            overall_late_probability,
            pred_id=pred_id,
            order_id=order["order_id"],
            order=order
        )

        # Persist prediction in SQLite
        database.save_order_prediction(
            internal_order_id=order["id"],
            prediction_id=pred_id,
            is_late=overall_is_late,
            late_probability=overall_late_probability,
            risk_label=risk_label,
            key_insight=result_data["key_insight"],
            recommendations_json=json.dumps(result_data["recommendations"])
        )

        # Clear any accumulated flash messages so messages never duplicate
        session.pop("_flashes", None)
        flash(f"Risk prediction completed for Order {order['order_id']}.", "success")
        return render_template("result.html", result=result_data, order=order)

    except Exception:
        app.logger.exception("Order prediction failed.")
        flash("Could not run delivery risk prediction on order.", "error")
        return redirect(url_for("home", lookup_id=order_id))


@app.route("/prediction-explanation/<order_id>", methods=["GET", "POST"])
@manager_required
def prediction_explanation(order_id):
    order = database.get_order_by_order_id(order_id)
    if not order:
        flash(f"Order '{order_id}' not found.", "error")
        return redirect(url_for("home"))

    items = order.get("items", [])
    if not items:
        flash("Order has no associated items to evaluate.", "error")
        return redirect(url_for("home", lookup_id=order_id))

    try:
        order_datetime = pd.to_datetime(f"{order['order_date']} {order['order_time']}:00")
        day_of_week = int(order_datetime.dayofweek)
        order_hour = int(order_datetime.hour)

        item_rows = []
        for item in items:
            row = {
                "Shipping Mode": order["shipping_mode"],
                "Days for shipment (scheduled)": float(order["days_scheduled"]),
                "Order Item Quantity": float(item["quantity"]),
                "Market": to_model_geo(order["market"]),
                "Order Region": to_model_geo(order["order_region"]),
                "Order Country": to_model_geo(order["order_country"]),
                "Order State": to_model_geo(order["order_state"]),
                "Order Item Product Price": float(item["product_price"]),
                "Order Item Discount Rate": float(item["discount_rate"]),
                "Sales": float(item["sales"]),
                "Type": order["order_type"],
                "Order Month": int(order_datetime.month),
                "Order Day": int(order_datetime.day),
                "Order Day of Week": day_of_week,
                "Order Hour": order_hour,
                "Is Weekend": int(day_of_week in [5, 6]),
                "Order Week of Year": float(order_datetime.isocalendar().week),
                "Is Night": int(order_hour < 6 or order_hour >= 22),
                "Interaction_ShippingMode_OrderHour": f"{order['shipping_mode']}_{order_hour}",
            }
            item_rows.append(row)

        df_items = pd.DataFrame(item_rows, columns=FEATURES)
        probs = model.predict_proba(df_items)
        classes = list(model.classes_)
        class_1_idx = classes.index(1)

        item_probs = [float(probs[i, class_1_idx]) for i in range(len(items))]
        max_idx = item_probs.index(max(item_probs))
        overall_late_probability = item_probs[max_idx]

        pred_id = f"SG-XAI-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

        result_data = build_prediction_result(
            item_rows[max_idx],
            overall_late_probability,
            pred_id=pred_id,
            order_id=order["order_id"],
            order=order
        )

        return render_template("prediction_explanation.html", result=result_data, order=order)

    except Exception:
        app.logger.exception("Prediction explanation failed.")
        flash("Could not generate prediction explanation for order.", "error")
        return redirect(url_for("home", lookup_id=order_id))


@app.route("/prediction-explanation", methods=["GET", "POST"])
@manager_required
def prediction_explanation_direct():
    order_id = request.args.get("order_id")
    if order_id:
        return redirect(url_for("prediction_explanation", order_id=order_id))
    recent = database.get_recent_orders(limit=1)
    if recent:
        return redirect(url_for("prediction_explanation", order_id=recent[0]["order_id"]))
    flash("No orders available to explain.", "error")
    return redirect(url_for("home"))


@app.route("/api/geo-hierarchy", methods=["GET"])
def get_geo_hierarchy():
    """Return the cascading geographic hierarchy as JSON."""
    return geo_hierarchy


@app.route("/result")
@manager_required
def result():
    return redirect(url_for("home"))


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
