import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# 2. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

HISTORY_FILE = BASE_DIR / "prediction_history.csv"


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

possible_paths = [
    BASE_DIR / "models" / "tuned_decision_tree_pipeline.pkl",
    BASE_DIR.parent / "models" / "tuned_decision_tree_pipeline.pkl",
]

MODEL_PATH = next(
    (path for path in possible_paths if path.exists()),
    None
)

if MODEL_PATH is None:
    st.error(
        "Model file not found. Please place "
        "'tuned_decision_tree_pipeline.pkl' inside your models folder."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --------------------------------------------------
# 4. SESSION STATE - PERMANENT PREDICTION HISTORY
# --------------------------------------------------

if "prediction_history" not in st.session_state:

    if HISTORY_FILE.exists():

        try:
            saved_history = pd.read_csv(HISTORY_FILE)

            st.session_state.prediction_history = (
                saved_history.to_dict(orient="records")
            )

        except Exception as e:

            st.error(
                f"Could not load prediction history: {e}"
            )

            st.session_state.prediction_history = []

    else:

        st.session_state.prediction_history = []


# --------------------------------------------------
# 5. CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #FFFFFF;
    }

    div[data-testid="stMetric"] {
        background-color: #1A1D24;
        border: 1px solid #292D36;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    div[data-testid="stMetricLabel"] {
        color: #B8C1CC;
    }

    div.stButton > button {
        background-color: #6C63FF;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #574FE0;
        color: white;
    }

    div[data-testid="stForm"] {
        background-color: #151922;
        border: 1px solid #292D36;
        padding: 20px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 6. TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Predict whether a telecom customer is likely to leave the company
    using a trained machine learning model.
    """
)

st.markdown("---")


# --------------------------------------------------
# 7. MODEL PERFORMANCE DASHBOARD
# --------------------------------------------------

st.header("📊 Model Performance Dashboard")

st.caption(
    "Performance metrics calculated on the test dataset."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Accuracy",
        value="74.59%"
    )

with col2:
    st.metric(
        label="Precision",
        value="51.40%"
    )

with col3:
    st.metric(
        label="Recall",
        value="78.34%"
    )

with col4:
    st.metric(
        label="F1-Score",
        value="62.08%"
    )


# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

st.markdown("---")

st.header("🔍 Confusion Matrix")

st.caption(
    "Confusion matrix results from the test dataset."
)

cm = [
    [758, 277],
    [81, 293]
]

fig, ax = plt.subplots(figsize=(7, 5))

matrix_image = ax.imshow(
    cm,
    interpolation="nearest"
)

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            str(cm[i][j]),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="black"
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Stay", "Churn"])
ax.set_yticklabels(["Stay", "Churn"])

ax.set_xlabel(
    "Predicted Customer",
    fontsize=12
)

ax.set_ylabel(
    "Actual Customer",
    fontsize=12
)

ax.set_title(
    "Customer Churn Confusion Matrix",
    fontsize=14
)

fig.colorbar(
    matrix_image,
    ax=ax
)

st.pyplot(fig)

plt.close(fig)


with st.expander("Understand the Confusion Matrix"):

    st.markdown(
        """
        **758 - True Negative (TN)**  
        The model correctly predicted customers who stayed.

        **277 - False Positive (FP)**  
        The model predicted churn, but these customers actually stayed.

        **81 - False Negative (FN)**  
        The model predicted stay, but these customers actually churned.

        **293 - True Positive (TP)**  
        The model correctly predicted customers who churned.
        """
    )


# --------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------

st.markdown("---")

st.header("⭐ Feature Importance")

st.caption(
    "The chart shows which features the trained Decision Tree "
    "used most when making predictions."
)

try:

    # Get the final estimator from the pipeline
    if hasattr(model, "named_steps"):

        final_estimator = list(
            model.named_steps.values()
        )[-1]

    else:

        final_estimator = model


    # Check feature importance availability
    if hasattr(final_estimator, "feature_importances_"):

        importances = final_estimator.feature_importances_

        feature_names = None


        # Get transformed feature names from preprocessing
        if hasattr(model, "named_steps"):

            try:

                preprocessing_pipeline = model[:-1]

                feature_names = (
                    preprocessing_pipeline.get_feature_names_out()
                )

            except Exception:

                feature_names = None


        # Fallback feature names
        if (
            feature_names is None
            or len(feature_names) != len(importances)
        ):

            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(importances))
            ]


        # Create feature importance dataframe
        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        )


        # Sort by importance
        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )


        # Display top 15 features
        top_features = importance_df.head(15)


        # Create chart
        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")

        ax.set_title(
            "Top 15 Features Influencing Customer Churn"
        )

        ax.invert_yaxis()

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        st.info(
            "Higher importance means the feature contributed more "
            "to the Decision Tree's predictions. Importance does "
            "not necessarily mean the feature causes churn."
        )

    else:

        st.warning(
            "Feature importance is not available for this model."
        )

except Exception as e:

    st.warning(
        f"Could not display feature importance: {e}"
    )


# --------------------------------------------------
# 10. CUSTOMER INPUT FORM
# --------------------------------------------------

st.markdown("---")

st.header("👤 Customer Information")

with st.form("customer_prediction_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col2:
        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col3:
        Partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        Dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col3:
        PhoneService = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    st.markdown("---")

    st.subheader("Services Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    with col2:
        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col3:
        OnlineSecurity = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        OnlineBackup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        DeviceProtection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        TechSupport = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    col1, col2 = st.columns(2)

    with col1:
        StreamingTV = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.markdown("---")

    st.subheader("Account Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        Contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        MonthlyCharges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col2:
        TotalCharges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=1.0
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# --------------------------------------------------
# 11. PREDICTION
# --------------------------------------------------

if submitted:

    customer_data = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [SeniorCitizen],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "tenure": [tenure],
            "PhoneService": [PhoneService],
            "MultipleLines": [MultipleLines],
            "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity],
            "OnlineBackup": [OnlineBackup],
            "DeviceProtection": [DeviceProtection],
            "TechSupport": [TechSupport],
            "StreamingTV": [StreamingTV],
            "StreamingMovies": [StreamingMovies],
            "Contract": [Contract],
            "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges]
        }
    )


    # Predict customer class
    prediction = int(
        model.predict(customer_data)[0]
    )


    # Get prediction probabilities
    probabilities = model.predict_proba(
        customer_data
    )[0]


    # Find churn probability
    churn_class_index = list(
        model.classes_
    ).index(1)

    churn_probability = float(
        probabilities[churn_class_index]
    )

    stay_probability = 1 - churn_probability


    # --------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------

    if churn_probability < 0.30:

        risk_level = "Low Risk 🟢"

    elif churn_probability < 0.60:

        risk_level = "Medium Risk 🟡"

    else:

        risk_level = "High Risk 🔴"


    # --------------------------------------------------
    # SAVE PREDICTION HISTORY
    # --------------------------------------------------

    prediction_record = {
        "Customer": len(
            st.session_state.prediction_history
        ) + 1,

        "Date & Time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),

        "Churn Probability": (
            f"{churn_probability * 100:.2f}%"
        ),

        "Stay Probability": (
            f"{stay_probability * 100:.2f}%"
        ),

        "Risk Level": risk_level,

        "Prediction": (
            "Churn" if prediction == 1 else "Stay"
        )
    }


    # Add record to session history
    st.session_state.prediction_history.append(
        prediction_record
    )


    # Save history permanently to CSV
    try:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        history_df.to_csv(
            HISTORY_FILE,
            index=False
        )

    except Exception as e:

        st.error(
            f"Could not save prediction history: {e}"
        )


    # --------------------------------------------------
    # DISPLAY PREDICTION RESULTS
    # --------------------------------------------------

    st.markdown("---")

    st.header("📈 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Stay Probability",
            f"{stay_probability * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Customer Risk Level",
            risk_level
        )


    # --------------------------------------------------
    # PROBABILITY BREAKDOWN
    # --------------------------------------------------

    st.subheader("Probability Breakdown")

    st.write(
        f"🔴 Churn Probability: "
        f"{churn_probability * 100:.2f}%"
    )

    st.progress(churn_probability)

    st.write(
        f"🟢 Stay Probability: "
        f"{stay_probability * 100:.2f}%"
    )

    st.progress(stay_probability)


    # --------------------------------------------------
    # FINAL PREDICTION MESSAGE
    # --------------------------------------------------

    st.markdown("---")

    if prediction == 1:

        st.error(
            "⚠️ This customer is likely to churn."
        )

        st.markdown(
            """
            **Recommendation:**  
            The company may consider contacting this customer
            and offering suitable retention benefits.
            """
        )

    else:

        st.success(
            "✅ This customer is likely to stay."
        )

        st.markdown(
            """
            **Recommendation:**  
            Continue providing good service and maintain
            customer satisfaction.
            """
        )


# --------------------------------------------------
# 12. PREDICTION HISTORY
# --------------------------------------------------

st.markdown("---")

st.header("📋 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Total Customers Analysed",
        len(st.session_state.prediction_history)
    )


    # --------------------------------------------------
    # DOWNLOAD HISTORY AS CSV
    # --------------------------------------------------

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="customer_churn_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )


    # --------------------------------------------------
    # CLEAR HISTORY
    # --------------------------------------------------

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        # Delete saved history file
        if HISTORY_FILE.exists():

            HISTORY_FILE.unlink()

        st.rerun()

else:

    st.info(
        "No predictions yet. Predict a customer to see the history."
    )


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)