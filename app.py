import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

THRESHOLD = 0.55
MODEL_FILE = "final_xgb_churn_model.json"
DATA_FILE = "Dataset.csv"
PLOTS_DIR = "churn_plots"


# ============================================================
# NOTEBOOK RESULTS
# These are displayed directly from the notebook results.
# No model retraining or test split is performed in Streamlit.
# ============================================================
MODEL_METRICS = {
    "Logistic Regression": [72.4236, 0.4885, 0.7968, 0.6057, 0.8342],
    "Decision Tree": [70.4335, 0.4666, 0.7834, 0.5848, 0.8175],
    "Random Forest": [74.6979, 0.5157, 0.7888, 0.6237, 0.8364],
    "XGBoost": [73.4186, 0.5000, 0.8182, 0.6207, 0.8407],
    "LightGBM": [72.9922, 0.4951, 0.8075, 0.6138, 0.8393],
}

THRESHOLD_RESULTS = pd.DataFrame({
    "Threshold": [
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ],
    "Precision": [
        0.350810, 0.369919, 0.380192, 0.403670, 0.422983, 0.443559,
        0.455319, 0.472727, 0.500000, 0.531646, 0.562762, 0.604938,
        0.632948, 0.659420, 0.686486, 0.788462, 0.900000
    ],
    "Recall": [
        0.983957, 0.973262, 0.954545, 0.941176, 0.925134, 0.893048,
        0.858289, 0.834225, 0.818182, 0.786096, 0.719251, 0.655080,
        0.585561, 0.486631, 0.339572, 0.219251, 0.096257
    ],
    "F1-Score": [
        0.517217, 0.536082, 0.543793, 0.565008, 0.580537, 0.592724,
        0.594995, 0.603482, 0.620690, 0.634304, 0.631455, 0.629012,
        0.608333, 0.560000, 0.454383, 0.343096, 0.173913
    ],
})


# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #111827, #1f2937);
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin-bottom: 0.4rem;
            font-size: 2.5rem;
        }

        .hero p {
            color: #d1d5db;
            font-size: 1.05rem;
        }

        .risk-high {
            padding: 1.2rem;
            border-radius: 14px;
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #7f1d1d;
        }

        .risk-high h2,
        .risk-high p {
            color: #7f1d1d !important;
        }

        .risk-low {
            padding: 1.2rem;
            border-radius: 14px;
            background: #dcfce7;
            border: 1px solid #bbf7d0;
            color: #166534;
        }

        .risk-low h2,
        .risk-low p {
            color: #166534 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PREPROCESSING REFERENCE
# This creates the SAME feature structure used during training.
# It does NOT train or split the data.
# ============================================================
@st.cache_data
def load_reference_features():
    df = pd.read_csv(DATA_FILE)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # Remove analysis-only columns if present.
    for col in [
        "MonthlyChargesGroup",
        "TenureGroup",
        "TotalChargesGroup",
    ]:
        if col in df.columns:
            df = df.drop(columns=col)

    df = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore",
    )

    # Feature engineering used by the notebook.
    df["Family"] = (
        (df["Partner"] == "Yes")
        & (df["Dependents"] == "Yes")
    ).astype(int)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["TotalServices"] = (
        df[service_cols] == "Yes"
    ).sum(axis=1)

    # Binary encoding.
    for col in [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]:
        df[col] = (
            df[col]
            .map({"Yes": 1, "No": 0})
            .astype(int)
        )

    dummy_columns = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    df = pd.get_dummies(
        df,
        columns=dummy_columns,
        drop_first=True,
        dtype=int,
    )

    return df


# ============================================================
# LOAD SAVED FINAL MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_FILE)
    return model


try:
    X_reference = load_reference_features()
    model = load_model()
except FileNotFoundError as e:
    missing = MODEL_FILE if MODEL_FILE not in [DATA_FILE] else DATA_FILE

    if not __import__("os").path.exists(DATA_FILE):
        st.error(
            f"{DATA_FILE} was not found. Put it in the same folder as app.py."
        )
    elif not __import__("os").path.exists(MODEL_FILE):
        st.error(
            f"{MODEL_FILE} was not found. Save your tuned XGBoost model "
            f"from the notebook first."
        )

    st.stop()


# ============================================================
# FEATURE NAMES
# ============================================================
FEATURE_LABELS = {
    "gender_Male": "Male",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Tenure": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines_No phone service": "No phone service",
    "MultipleLines_Yes": "Multiple lines",
    "InternetService_Fiber optic": "Fiber optic internet",
    "InternetService_No": "No internet service",
    "OnlineSecurity_Yes": "Online security",
    "OnlineSecurity_No internet service": "No internet service / security",
    "OnlineBackup_Yes": "Online backup",
    "OnlineBackup_No internet service": "No internet service / backup",
    "DeviceProtection_Yes": "Device protection",
    "DeviceProtection_No internet service": "No internet service / protection",
    "TechSupport_Yes": "Tech support",
    "TechSupport_No internet service": "No internet service / support",
    "StreamingTV_Yes": "Streaming TV",
    "StreamingTV_No internet service": "No internet service / TV",
    "StreamingMovies_Yes": "Streaming movies",
    "StreamingMovies_No internet service": "No internet service / movies",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod_Credit card (automatic)": "Credit card (automatic)",
    "PaymentMethod_Electronic check": "Electronic check",
    "PaymentMethod_Mailed check": "Mailed check",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "Family": "Family",
    "TotalServices": "Total services",
}


def friendly_name(feature):
    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " "),
    )


# ============================================================
# CUSTOMER INPUT PREPROCESSING
# ============================================================
def build_input(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    row["Family"] = (
        (row["Partner"] == "Yes")
        & (row["Dependents"] == "Yes")
    ).astype(int)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    row["TotalServices"] = (
        row[service_cols] == "Yes"
    ).sum(axis=1)

    for col in [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]:
        row[col] = (
            row[col]
            .map({"Yes": 1, "No": 0})
            .astype(int)
        )

    dummy_columns = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    row = pd.get_dummies(
        row,
        columns=dummy_columns,
        drop_first=True,
        dtype=int,
    )

    return row


def align_features(row):
    return row.reindex(
        columns=X_reference.columns,
        fill_value=0,
    ).astype(float)


def get_shap_values(input_row):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_row)

    values = np.asarray(explanation.values)

    if values.ndim == 3:
        values = values[:, :, -1]

    return explanation, values[0]


def render_contribution_lists(shap_row, input_row):
    contrib = pd.DataFrame({
        "feature": input_row.columns,
        "shap": shap_row,
    })

    contrib["label"] = contrib["feature"].map(
        friendly_name
    )

    positive = (
        contrib[contrib["shap"] > 0]
        .sort_values("shap", ascending=False)
        .head(5)
    )

    negative = (
        contrib[contrib["shap"] < 0]
        .sort_values("shap", ascending=True)
        .head(5)
    )

    return positive, negative


# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.title("Customer Churn")
st.sidebar.caption("XGBoost • SHAP • Threshold Optimization")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Churn Prediction",
        "🔍 Prediction Explanation",
        "📊 Model Insights",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    f"Prediction threshold: **{THRESHOLD:.2f}**"
)
st.sidebar.caption("Final model: **XGBoost**")


# ============================================================
# OVERVIEW
# ============================================================
if page == "🏠 Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Prediction</h1>
            <p>
                An explainable machine-learning application that predicts
                customer churn probability using a tuned XGBoost model and SHAP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Final Model", "XGBoost")
    c2.metric("ROC-AUC", "0.8407")
    c3.metric("Threshold", "0.55")
    c4.metric("Recall @ 0.55", "78.61%")

    st.subheader("What this project demonstrates")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 🤖 Machine Learning")
        st.write(
            "Customer-level churn probability prediction using the tuned XGBoost model."
        )

    with b:
        st.markdown("### 🔍 Explainability")
        st.write(
            "SHAP explains which customer features push an individual prediction toward or away from churn."
        )

    with c:
        st.markdown("### 🎯 Threshold Optimization")
        st.write(
            "The deployment decision uses the selected 0.55 threshold instead of the default 0.50."
        )

    st.divider()

    st.subheader("How the app works")

    st.markdown(
        """
        **Enter customer details → Predict churn probability → Apply 0.55 threshold → Explain the prediction**
        """
    )

    st.info(
        "The final Streamlit app loads the trained XGBoost model saved from the notebook. "
        "It does not retrain or split the dataset."
    )


# ============================================================
# CHURN PREDICTION
# ============================================================
elif page == "🔮 Churn Prediction":

    st.title("🔮 Churn Prediction")
    st.write(
        "Enter customer information and generate an individual churn prediction."
    )

    with st.form("prediction_form"):

        st.subheader("Customer information")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

        with c2:
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
            )

        with c3:
            partner = st.selectbox(
                "Partner",
                ["No", "Yes"],
            )

        with c4:
            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"],
            )

        st.subheader("Services")

        c1, c2, c3 = st.columns(3)

        with c1:
            tenure = st.number_input(
                "Tenure (months)",
                0,
                72,
                12,
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "No phone service",
                    "No",
                    "Yes",
                ],
            )

        with c2:
            internet_service = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No",
                ],
            )

            online_security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

        with c3:
            device_protection = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
            )

        st.subheader("Account")

        c1, c2, c3 = st.columns(3)

        with c1:
            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
            )

        with c2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with c3:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=0.01,
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=840.0,
                step=0.01,
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        raw_input = build_input(
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        )

        input_row = align_features(raw_input)

        probability = float(
            model.predict_proba(input_row)[0, 1]
        )

        prediction = int(
            probability >= THRESHOLD
        )

        st.session_state["last_input"] = input_row
        st.session_state["last_probability"] = probability
        st.session_state["last_prediction"] = prediction

        st.divider()
        st.subheader("Prediction")

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.1f}%",
            )

        with r2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.2f}",
            )

        if prediction == 1:

            st.markdown(
                f"""
                <div class="risk-high">
                    <h2>⚠️ Predicted Churn: YES</h2>
                    <p>
                        The predicted probability ({probability:.1%}) is at or above
                        the deployment threshold of {THRESHOLD:.2f}.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="risk-low">
                    <h2>✓ Predicted Churn: NO</h2>
                    <p>
                        The predicted probability ({probability:.1%}) is below
                        the deployment threshold of {THRESHOLD:.2f}.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(min(probability, 1.0))

        st.info(
            "Go to **Prediction Explanation** to see which customer features "
            "contributed most to this prediction."
        )


# ============================================================
# PREDICTION EXPLANATION
# ============================================================
elif page == "🔍 Prediction Explanation":

    st.title("🔍 Prediction Explanation")

    if "last_input" not in st.session_state:

        st.info(
            "Make a prediction first on the **Churn Prediction** page."
        )

    else:

        input_row = st.session_state["last_input"]
        probability = st.session_state["last_probability"]

        st.metric(
            "Current Churn Probability",
            f"{probability * 100:.1f}%",
        )

        explanation, shap_row = get_shap_values(
            input_row
        )

        positive, negative = render_contribution_lists(
            shap_row,
            input_row,
        )

        st.subheader("Why is this customer at risk?")

        left, right = st.columns(2)

        with left:

            st.markdown(
                "### Factors increasing churn risk"
            )

            if positive.empty:
                st.write(
                    "No positive SHAP contributions were identified."
                )
            else:

                for _, row in positive.iterrows():

                    st.write(
                        f"• **{row['label']}** "
                        f"(+{row['shap']:.3f})"
                    )

        with right:

            st.markdown(
                "### Factors reducing churn risk"
            )

            if negative.empty:
                st.write(
                    "No negative SHAP contributions were identified."
                )
            else:

                for _, row in negative.iterrows():

                    st.write(
                        f"• **{row['label']}** "
                        f"({row['shap']:.3f})"
                    )

        st.divider()

        st.subheader("SHAP Waterfall Plot")

        try:

            fig = plt.figure(figsize=(10, 6))

            shap.plots.waterfall(
                explanation[0],
                max_display=12,
                show=False,
            )

            st.pyplot(
                fig,
                clear_figure=True,
            )

            plt.close(fig)

        except Exception:

            st.warning(
                "The SHAP waterfall plot could not be rendered with "
                "the installed SHAP version."
            )

        st.caption(
            "Positive SHAP values push the prediction toward churn; "
            "negative values push it away from churn."
        )


# ============================================================
# MODEL INSIGHTS
# No retraining. No dataset split.
# Uses results already obtained in the notebook.
# ============================================================
elif page == "📊 Model Insights":

    st.title("📊 Model Insights")

    st.subheader("Tuned Model Performance")

    performance = pd.DataFrame(
        MODEL_METRICS,
        index=[
            "Accuracy (%)",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
        ],
    ).T

    st.dataframe(
        performance.style.format({
            "Accuracy (%)": "{:.2f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
        }),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Tuned Model Performance Comparison")
    st.image(
        f"{PLOTS_DIR}/tuned_model_performance.png",
        caption="Tuned Model Performance Comparison",
        use_container_width=True,
    )

    st.divider()

    st.subheader("ROC-AUC Before vs After Tuning")
    st.image(
        f"{PLOTS_DIR}/roc_auc_before_after.png",
        caption="ROC-AUC Before vs After Tuning",
        use_container_width=True,
    )

    st.subheader("Accuracy Before vs After Tuning")
    st.image(
        f"{PLOTS_DIR}/accuracy_before_after.png",
        caption="Accuracy Before vs After Tuning",
        use_container_width=True,
    )

    st.divider()

    st.subheader("XGBoost Feature Importance")
    st.image(
        f"{PLOTS_DIR}/xgb_feature_importance.png",
        caption="Top 15 Features - XGBoost",
        use_container_width=True,
    )

    st.divider()

    st.subheader("SHAP Feature Importance")
    st.image(
        f"{PLOTS_DIR}/shap_feature_importance.png",
        caption="SHAP Feature Importance",
        use_container_width=True,
    )

    st.subheader("SHAP Beeswarm Plot")
    st.image(
        f"{PLOTS_DIR}/shap_beeswarm.png",
        caption="SHAP Beeswarm Plot",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Threshold Optimization")

    st.image(
        f"{PLOTS_DIR}/threshold_optimization.png",
        caption="Threshold Optimization - XGBoost",
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Selected Threshold", "0.55")
    c2.metric("Precision @ 0.55", "0.5316")
    c3.metric("Recall @ 0.55", "0.7861")

    st.metric("F1-Score @ 0.55", "0.6343")

    st.divider()

    st.subheader("Final Confusion Matrix")

    st.image(
        f"{PLOTS_DIR}/confusion_matrix.png",
        caption="Final Confusion Matrix - XGBoost (Threshold = 0.55)",
        use_container_width=True,
    )

    st.info(
        "These global model-insight visualizations are the actual plots "
        "generated in the project notebook. They are displayed as saved "
        "artifacts, so Streamlit does not retrain or re-evaluate the models."
    )

    st.divider()

    st.subheader("Deployment Pipeline")

    st.code(
        """
Notebook
    ↓
Hyperparameter tuning
    ↓
tuned_xgb
    ↓
final_xgb_churn_model.json
    ↓
Streamlit loads saved model
    ↓
Customer input
    ↓
Same preprocessing
    ↓
XGBoost predict_proba()
    ↓
Churn probability
    ↓
Threshold = 0.55
    ↓
YES / NO
    ↓
Individual SHAP explanation
        """,
        language="text",
    )
