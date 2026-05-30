import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# =====================================================
# TITLE
# =====================================================

st.title("📡 AI-Powered Telecom Customer Churn Prediction Platform")

st.write(
    "Predict customer churn probability, risk level, "
    "customer segment and revenue impact."
)

# =====================================================
# INPUT SECTION
# =====================================================

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        1,
        72,
        12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        value=1500.0
    )

# =====================================================
# MANUAL ENCODING
# =====================================================

gender = 0 if gender == "Female" else 1

partner = 1 if partner == "Yes" else 0

dependents = 1 if dependents == "Yes" else 0

phone_service = 1 if phone_service == "Yes" else 0

online_security = 1 if online_security == "Yes" else 0

tech_support = 1 if tech_support == "Yes" else 0

streaming_tv = 1 if streaming_tv == "Yes" else 0

paperless = 1 if paperless == "Yes" else 0

# Internet Service Encoding
internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

internet_service = internet_map[internet_service]

# Contract Encoding
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

contract = contract_map[contract]

# Payment Encoding
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

payment_method = payment_map[payment_method]

# =====================================================
# FEATURE ARRAY
# =====================================================

features = np.array([[
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone_service,
    0,
    internet_service,
    online_security,
    0,
    0,
    tech_support,
    streaming_tv,
    0,
    contract,
    paperless,
    payment_method,
    monthly_charges,
    total_charges
]])

# =====================================================
# SCALING
# =====================================================

features_scaled = scaler.transform(features)

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Churn"):

    prediction = model.predict(features_scaled)[0]

    probability = model.predict_proba(
        features_scaled
    )[0][1] * 100

    # =================================================
    # RISK LEVEL
    # =================================================

    if probability >= 75:
        risk = "HIGH"

    elif probability >= 40:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    # =================================================
    # CUSTOMER SEGMENT
    # =================================================

    if tenure >= 48:
        segment = "Loyal Customer"

    elif monthly_charges > 80:
        segment = "Premium Customer"

    else:
        segment = "At-Risk Customer"

    # =================================================
    # REVENUE LOSS
    # =================================================

    expected_loss = monthly_charges * 12

    # =================================================
    # RESULTS
    # =================================================

    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠ Customer likely to churn")

    else:
        st.success("✅ Customer likely to stay")

    st.write(
        "### Churn Probability:",
        round(probability, 2),
        "%"
    )

    st.write(
        "### Risk Level:",
        risk
    )

    st.write(
        "### Customer Segment:",
        segment
    )

    st.write(
        "### Expected Annual Revenue Loss: ₹",
        round(expected_loss, 2)
    )

    # =================================================
    # RETENTION RECOMMENDATION
    # =================================================

    st.subheader("Retention Recommendation")

    if risk == "HIGH":

        st.warning(
            "Offer discount plans, loyalty rewards, "
            "or premium support."
        )

    elif risk == "MEDIUM":

        st.info(
            "Provide engagement offers and "
            "personalized plans."
        )

    else:

        st.success(
            "Customer is stable. Maintain service quality."
        )

    # =================================================
    # GRAPH 1
    # =================================================

    st.subheader("Churn Probability Graph")

    fig1, ax1 = plt.subplots()

    ax1.bar(
        ["Stay", "Churn"],
        [100 - probability, probability]
    )

    ax1.set_ylabel("Probability")

    st.pyplot(fig1)

    # =================================================
    # GRAPH 2
    # =================================================

    st.subheader("Customer Charges")

    fig2, ax2 = plt.subplots()

    ax2.bar(
        ["Monthly Charges", "Yearly Revenue"],
        [monthly_charges, expected_loss]
    )

    st.pyplot(fig2)

    # =================================================
    # GRAPH 3
    # =================================================

    st.subheader("Customer Tenure")

    fig3, ax3 = plt.subplots()

    ax3.plot(
        [0, tenure],
        [0, monthly_charges]
    )

    ax3.set_xlabel("Tenure")

    ax3.set_ylabel("Monthly Charges")

    st.pyplot(fig3)