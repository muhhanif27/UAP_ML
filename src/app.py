import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Disaster Severity Prediction",
    layout="centered",
    page_icon="🌪️"
)

# ========================= LIGHT + ORANGE THEME =========================
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* Headings */
h1, h2, h3 {
    color: #f97316;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 2px solid #fed7aa;
}

/* Cards */
div[data-testid="stMetric"],
div[data-testid="stVerticalBlock"] > div {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #f97316, #fb923c);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #ea580c, #f97316);
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #f97316;
}
</style>
""", unsafe_allow_html=True)

# ========================= TITLE =========================
st.title("🌪️ Disaster Severity Class Prediction")
st.markdown(
    "Prediksi tingkat keparahan bencana: **Minor**, **Moderate**, atau **Severe**"
)

# ========================= PATH =========================
BASE_PATH = "models"

# ========================= LOAD PREPROCESSOR =========================
@st.cache_resource
def load_preprocessors():
    scaler = joblib.load(os.path.join(BASE_PATH, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(BASE_PATH, "label_encoders.pkl"))
    return scaler, label_encoders

scaler, label_encoders = load_preprocessors()

# ========================= LOAD MODELS =========================
@st.cache_resource
def load_mlp_model():
    return load_model(os.path.join(BASE_PATH, "mlp_disaster_model.keras"))

@st.cache_resource
def load_embedding_model():
    return load_model(os.path.join(BASE_PATH, "embedding_nn_disaster_model.keras"))

@st.cache_resource
def load_tabnet_model():
    model = TabNetClassifier()
    model.load_model(os.path.join(BASE_PATH, "tabnet_disaster_model.zip"))
    return model

# ========================= COLUMNS =========================
cat_cols = ["disaster_type", "location", "aid_provided"]
num_cols = [
    "latitude", "longitude", "severity_level",
    "affected_population", "estimated_economic_loss_usd",
    "response_time_hours", "infrastructure_damage_index"
]

class_names = ["Minor", "Moderate", "Severe"]

# ========================= PREPROCESS =========================
def preprocess_data(df_input):
    df = df_input.copy()

    for col in cat_cols:
        le = label_encoders[col]
        df[col] = df[col].map(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    X = df[cat_cols + num_cols]
    X[num_cols] = scaler.transform(X[num_cols])
    return X

# ========================= SIDEBAR =========================
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["MLP", "Embedding Neural Network", "TabNet"]
)

if model_choice == "MLP":
    model = load_mlp_model()
    st.sidebar.success("MLP loaded")
elif model_choice == "Embedding Neural Network":
    model = load_embedding_model()
    st.sidebar.success("Embedding NN loaded")
else:
    model = load_tabnet_model()
    st.sidebar.success("TabNet loaded")

# ========================= TABS =========================
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction (CSV)"])

# ========================= TAB 1 =========================
with tab1:
    st.subheader("Input Data Bencana")

    col1, col2 = st.columns(2)

    with col1:
        disaster_type = st.selectbox(
            "Disaster Type",
            ["Wildfire", "Hurricane", "Earthquake", "Flood", "Drought",
             "Volcanic Eruption", "Landslide"]
        )
        location = st.text_input("Location", "Indonesia")
        latitude = st.number_input("Latitude", value=-6.2088, format="%.6f")
        longitude = st.number_input("Longitude", value=106.8456, format="%.6f")
        severity_level = st.slider("Severity Level (1–10)", 1, 10, 5)

    with col2:
        affected_population = st.number_input("Affected Population", 0, value=10000)
        economic_loss = st.number_input(
            "Estimated Economic Loss (USD)", 0.0, value=5_000_000.0
        )
        response_time = st.number_input("Response Time (Hours)", 0.0, value=24.0)
        aid_provided = st.selectbox("Aid Provided", ["Yes", "No"])
        infra_damage = st.slider(
            "Infrastructure Damage Index", 0.0, 1.0, 0.5, step=0.01
        )

    if st.button("🚀 Prediksi Tingkat Keparahan", type="primary"):
        input_df = pd.DataFrame([{
            "disaster_type": disaster_type,
            "location": location,
            "latitude": latitude,
            "longitude": longitude,
            "severity_level": severity_level,
            "affected_population": affected_population,
            "estimated_economic_loss_usd": economic_loss,
            "response_time_hours": response_time,
            "aid_provided": aid_provided,
            "infrastructure_damage_index": infra_damage,
            "is_major_disaster": 0
        }])

        X = preprocess_data(input_df)

        if model_choice == "TabNet":
            probas = model.predict_proba(X.values)[0]
        elif model_choice == "Embedding Neural Network":
            embed_inputs = [X[c].values for c in cat_cols] + [X[num_cols].values]
            probas = model.predict(embed_inputs)[0]
        else:
            probas = model.predict(X)[0]

        pred = np.argmax(probas)

        st.success(f"🎯 **Prediksi: {class_names[pred]}**")
        st.metric("Confidence", f"{np.max(probas):.2%}")

        st.bar_chart(
            pd.DataFrame({
                "Class": class_names,
                "Probability": probas
            }).set_index("Class")
        )

# ========================= TAB 2 =========================
with tab2:
    st.subheader("Batch Prediction dari CSV")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.dataframe(df_upload.head())

        if st.button("⚡ Prediksi Batch"):
            X = preprocess_data(df_upload)

            if model_choice == "TabNet":
                probas = model.predict_proba(X.values)
            elif model_choice == "Embedding Neural Network":
                embed_inputs = [X[c].values for c in cat_cols] + [X[num_cols].values]
                probas = model.predict(embed_inputs)
            else:
                probas = model.predict(X)

            predictions = np.argmax(probas, axis=1)

            df_upload["Predicted_Severity_Class"] = [
                class_names[p] for p in predictions
            ]
            df_upload["Confidence"] = [
                f"{np.max(p):.2%}" for p in probas
            ]

            cols = [c for c in df_upload.columns
                    if c not in ["Predicted_Severity_Class", "Confidence"]] + \
                   ["Predicted_Severity_Class", "Confidence"]

            df_upload = df_upload[cols]

            st.success("✅ Batch prediction selesai")
            st.dataframe(df_upload)

            st.download_button(
                "⬇️ Download Hasil Prediksi",
                df_upload.to_csv(index=False).encode(),
                "prediction_results.csv",
                "text/csv"
            )

# ========================= FOOTER =========================
st.caption("Synthetic Disaster Dataset 2025 | Streamlit ML Dashboard")
