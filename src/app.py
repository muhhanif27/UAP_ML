import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch

from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier
from rtdl_revisiting_models import FTTransformer

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Disaster Severity Prediction",
    layout="centered",
    page_icon="🌪️"
)

# ========================= THEME =========================
st.markdown("""
<style>
.stApp { background-color: #f8fafc; color: #0f172a; }
h1, h2, h3 { color: #f97316; }
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 2px solid #fed7aa;
}
div[data-testid="stMetric"],
div[data-testid="stVerticalBlock"] > div {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
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
</style>
""", unsafe_allow_html=True)

# ========================= TITLE =========================
st.title("🌪️ Disaster Severity Class Prediction")
st.markdown("Prediksi tingkat keparahan bencana: **Minor**, **Moderate**, atau **Severe**")

# ========================= PATH & COLUMNS =========================
BASE_PATH = "models"

cat_cols = ["disaster_type", "location", "aid_provided"]
num_cols = [
    "latitude", "longitude", "severity_level",
    "affected_population", "estimated_economic_loss_usd",
    "response_time_hours", "infrastructure_damage_index"
]
class_names = ["Minor", "Moderate", "Severe"]

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
def load_tabnet_model():
    model = TabNetClassifier()
    model.load_model(os.path.join(BASE_PATH, "tabnet_disaster_model.zip"))
    return model

@st.cache_resource
def load_ft_transformer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FTTransformer(
        n_cont_features=len(num_cols),
        cat_cardinalities=[len(label_encoders[c].classes_) for c in cat_cols],
        d_block=32,
        n_blocks=4,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4/3,
        ffn_dropout=0.2,
        residual_dropout=0.1,
        d_out=3
    )
    state_dict = torch.load(
        os.path.join(BASE_PATH, "ft_transformer_disaster_model.pth"),
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# ========================= PREPROCESS =========================
def preprocess_data(df_input):
    df = df_input.copy()
    for col in cat_cols:
        le = label_encoders[col]
        # Jika nilai baru tidak ada di encoder, gunakan kelas pertama sebagai fallback
        df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    X = df[cat_cols + num_cols]
    X[num_cols] = scaler.transform(X[num_cols])
    return X

# ========================= SIDEBAR =========================
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["MLP", "TabNet", "FT-Transformer"]
)

if model_choice == "MLP":
    model = load_mlp_model()
    device = None  # Tidak dipakai untuk MLP
elif model_choice == "TabNet":
    model = load_tabnet_model()
    device = None
else:  # FT-Transformer
    model, device = load_ft_transformer()

st.sidebar.success(f"✅ {model_choice} loaded")

# ========================= TABS =========================
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction (CSV)"])

# ========================= TAB 1: SINGLE PREDICTION =========================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        disaster_type = st.selectbox(
            "Disaster Type",
            ["Wildfire", "Hurricane", "Earthquake", "Flood", "Drought",
             "Volcanic Eruption", "Landslide"]
        )
        location = st.text_input("Location", "Indonesia")
        latitude = st.number_input("Latitude", value=-6.2)
        longitude = st.number_input("Longitude", value=106.8)
        severity_level = st.slider("Severity Level", 1, 10, 5)

    with col2:
        affected_population = st.number_input("Affected Population", value=10000)
        economic_loss = st.number_input("Economic Loss (USD)", value=5_000_000.0)
        response_time = st.number_input("Response Time (Hours)", value=24.0)
        aid_provided = st.selectbox("Aid Provided", ["Yes", "No"])
        infra_damage = st.slider("Infrastructure Damage Index", 0.0, 1.0, 0.5)

    if st.button("🚀 Predict"):
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
            "infrastructure_damage_index": infra_damage
        }])

        X = preprocess_data(input_df)

        # === Get probabilities ===
        if model_choice == "TabNet":
            probas = model.predict_proba(X.values)[0]
        elif model_choice == "FT-Transformer":
            num_tensor = torch.tensor(X[num_cols].values, dtype=torch.float32).to(device)
            cat_tensor = torch.tensor(X[cat_cols].values, dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(num_tensor, cat_tensor)
                probas = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:  # MLP
            probas = model.predict(X.values)[0]  # Keras model predict langsung ke X

        probas = np.array(probas, dtype=float)
        pred_idx = np.argmax(probas)
        pred_class = class_names[pred_idx]

        st.success(f"🎯 **Prediksi: {pred_class}**")

        st.markdown("### 📊 Probability Distribution")

        cols = st.columns(3)
        for i, (col, cls_name) in enumerate(zip(cols, class_names)):
            prob = probas[i]
            with col:
                if i == pred_idx:
                    st.markdown(f"**{cls_name}** 🌟")
                else:
                    st.markdown(f"**{cls_name}**")
                st.progress(prob)
                st.write(f"**{prob:.2%}**")

        st.info(f"💡 Confidence tertinggi: **{pred_class}** dengan {np.max(probas):.2%}")

# ========================= TAB 2: BATCH PREDICTION =========================
with tab2:
    uploaded_file = st.file_uploader("Upload CSV file dengan kolom yang sesuai", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview data:")
        st.dataframe(df.head())

        if st.button("⚡ Predict Batch"):
            X = preprocess_data(df)

            # === Get probabilities ===
            if model_choice == "TabNet":
                probas = model.predict_proba(X.values)
            elif model_choice == "FT-Transformer":
                num_tensor = torch.tensor(X[num_cols].values, dtype=torch.float32).to(device)
                cat_tensor = torch.tensor(X[cat_cols].values, dtype=torch.long).to(device)
                with torch.no_grad():
                    logits = model(num_tensor, cat_tensor)
                    probas = torch.softmax(logits, dim=1).cpu().numpy()
            else:  # MLP
                probas = model.predict(X.values)

            probas = np.array(probas, dtype=float)
            predictions = np.argmax(probas, axis=1)
            df["Predicted_Severity"] = [class_names[p] for p in predictions]
            df["Confidence"] = [f"{np.max(p):.2%}" for p in probas]

            st.success("✅ Batch prediction selesai!")

            # === Distribusi Kelas ===
            st.markdown("### 📈 Distribusi Prediksi Kelas")
            class_counts = df["Predicted_Severity"].value_counts()

            cols = st.columns(3)
            colors = ["#10b981", "#f59e0b", "#ef4444"]  # Hijau, Kuning, Merah
            for i, cls_name in enumerate(class_names):
                count = class_counts.get(cls_name, 0)
                percentage = count / len(df) * 100
                with cols[i]:
                    st.metric(
                        label=cls_name,
                        value=count,
                        delta=f"{percentage:.1f}%"
                    )

            st.markdown("### 📋 Hasil Prediksi Lengkap")
            st.dataframe(df)

            # Opsional: download hasil
            csv = df.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Hasil Prediksi",
                csv,
                "predictions_disaster_severity.csv",
                "text/csv"
            )