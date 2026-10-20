# SupplyGuard: AI-Powered Supply Chain Late Delivery Risk Assessment Platform

SupplyGuard is an intelligent, end-to-end supply chain risk management platform designed to predict and mitigate shipment delivery delays before orders are dispatched. Powered by an optimized 19-feature Random Forest machine learning pipeline trained on the DataCo Global Supply Chain dataset, SupplyGuard delivers pre-dispatch delay predictions with **91.18% ROC-AUC** and **95.13% recall on delayed orders**.

---

## 🌟 Key Features

### 1. Distinct Portals with Role-Based Access Control (RBAC)
- **Customer Portal (`/order`)**:
  - Clean, dedicated e-commerce order placement experience.
  - **3-Step Checkout Wizard**:
    - **Step 1: Customer & Payment** (Customer details, payment method).
    - **Step 2: Order Items & Promotions** (Product selection from live catalogue, itemized quantity, and automated coupon code validation).
    - **Step 3: Shipping & Delivery** (Cascading geographic routing and shipping mode selection).
  - Instant order receipt generation and order tracking ID assignment.
  - Complete security separation: Customers cannot access manager dashboards or internal ML assessment tools.

- **Manager Operations Dashboard (`/`)**:
  - Live order search and retrieval by Order ID.
  - Recent order tracking feed with persistent SQLite logging.
  - **One-Click Risk Assessment**: Instant machine learning evaluation of pre-dispatch delivery risk.
  - Visual circular risk probability gauge (High Risk vs. Low Risk).
  - Root-cause explanation ("Key Insight") and actionable mitigation steps ("Recommendations").
  - **Manual 19-Feature Calculator (`/manual-assessment`)**: What-if scenario testing for logistics planners.

### 2. Cascading Geographic Routing Engine
- Cascading dropdowns dynamically filtered in real time:
  $$\text{Market} \longrightarrow \text{Order Region} \longrightarrow \text{Order Country} \longrightarrow \text{Order State}$$
- Validates geography consistency and translates clean display names to historical dataset nomenclature (`geo_mapping.json` & `geo_hierarchy.json`).

### 3. Promotional Coupon System
- Automatic coupon code verification:
  - `WELCOME10`: 10% Discount
  - `SAVE15`: 15% Discount
  - `SUPER20`: 20% Discount
  - `SUPPLYGUARD`: 15% VIP Discount

---

## 🤖 Machine Learning Model Architecture

The core prediction model is a **Random Forest Classifier** (200 estimators) trained on 19 selected pre-dispatch features available at order placement:

1. **Shipping & Transit**: `Shipping Mode`, `Days for shipment (scheduled)`, `Interaction_ShippingMode_OrderHour`
2. **Customer & Payment**: `Type` (Payment method: DEBIT, TRANSFER, PAYMENT, CASH)
3. **Geography & Location**: `Market`, `Order Region`, `Order Country`, `Order State`
4. **Order Economics**: `Order Item Product Price`, `Order Item Discount Rate`, `Order Item Quantity`, `Sales`
5. **Temporal & Seasonal**: `Order Month`, `Order Day`, `Order Day of Week`, `Order Hour`, `Is Weekend`, `Order Week of Year`, `Is Night`

### Performance Benchmarks (Evaluated on 36,104 Test Records)
- **ROC-AUC**: **91.18%**
- **Decision Threshold**: **0.41** (Optimized for operational sensitivity)
- **Late Delivery Recall**: **95.13%** (Catches over 95 out of 100 delayed shipments)
- **Overall Accuracy**: **83.89%**

---

## 📁 Project Structure

```
SupplyGuard_19_Features/
├── app.py                               # Flask backend application & routing
├── database.py                          # SQLite database schema, orders & catalogue
├── supplyguard.db                       # SQLite persistent storage
├── train_model.py                       # 19-feature training pipeline script
├── users.json                           # Role-based user credentials & hashes
├── requirements.txt                     # Python package dependencies
├── README.md                            # Project documentation
├── DataCoSupplyChainDataset.csv         # Historical training & evaluation dataset
├── DescriptionDataCoSupplyChain.csv     # Dataset column definitions
├── results/
│   ├── metrics.json                     # Comprehensive model evaluation benchmarks
│   └── metrics.csv
├── saved_model/
│   ├── supplyguard_19_model.joblib      # Serialized Random Forest model
│   ├── model_config.json                # Feature list & decision threshold
│   ├── geo_hierarchy.json               # Nested geographic relationship tree
│   └── geo_mapping.json                 # Name translation dictionary
├── static/
│   ├── style.css                        # Glassmorphism design system & animations
│   └── img/
│       └── brand_shield.png             # SupplyGuard logo
└── templates/
    ├── home.html                        # Manager operations dashboard
    ├── login.html                       # Role-based portal sign-in
    ├── register.html                    # User registration
    ├── order.html                       # 3-step customer checkout wizard
    ├── order_confirmation.html          # Order receipt & confirmation
    ├── manual_assessment.html           # Manual 19-feature risk assessment
    └── result.html                      # Delivery risk assessment report
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### 2. Installation
Open a terminal in the project directory:

```bash
cd SupplyGuard_19_Features
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔑 Demo Login Accounts

Pre-configured accounts for testing:

| Role | Email | Password |
| :--- | :--- | :--- |
| **Manager** | `manager_demo@supplyguard.com` | `Password123!` |
| **Customer** | `customer_demo@supplyguard.com` | `Password123!` |

*(You can also click **Register** on the login page to create new accounts as a Customer or Manager).*

---

## 🔄 Retraining the Model

To retrain the 19-feature Random Forest pipeline from scratch using the dataset:

```bash
python train_model.py
```
This will automatically update the serialized model in `saved_model/` and log updated metrics to `results/`.
