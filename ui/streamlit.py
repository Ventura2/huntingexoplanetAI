import streamlit as st
import random
import time
import pandas as pd
from tabpfn import TabPFNClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

USE_MOCK=True
# Configuración inicial de la página
st.set_page_config(
    page_title="🌌 Exoplanet Prediction",
    page_icon="🪐",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #1e1b4b, #000, #3b0764);
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 Exoplanet Prediction (TabPfn AI classifier)")
st.write("Make a prediction using your Tabpfn model trained")

# --- Nuevo campo para subir fichero ---
model_file = st.file_uploader("📂 Choose the model file", type=["pkl"])

# Mostrar info del fichero si se subió uno
if model_file is not None:
    st.info(f"✅ File uploaded: `{model_file.name}`")
    st.write("File type:", model_file.type)
    st.write("File size:", f"{model_file.size / 1024:.2f} KB")

# --- Nuevo campo para subir fichero ---
csv_file = st.file_uploader("📂 Choose the csv file", type=["csv"])

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write("### Vista previa de tus datos")
    st.dataframe(df.head())
    
# --- Formulario principal ---
with st.form("exoplanet_form"):

    submitted = st.form_submit_button("🔮 Predict")

# --- Acción al enviar el formulario ---
if submitted:
    with st.spinner("Running prediction model..."):
        
        if USE_MOCK:
            # --- Mock predictions ---
            y_pred_encoded = np.random.randint(0, 3, size=len(df))  # 0,1,2
            le = LabelEncoder()
            le.classes_ = np.array(["CANDIDATE", "FALSE POSITIVE", "CONFIRMED"])
            y_pred = le.inverse_transform(y_pred_encoded)

            # Mock test accuracy
            y_test_mock = np.random.choice(le.classes_, size=len(df))
            test_accuracy = 0.92
            time.sleep(2)
            
        else:
            # Cargar después
            clf = joblib.load(model_file.name)
            # Keep only numeric and categorical relevant features
            features = [
                'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
                'koi_srad', 'koi_smass', 'koi_impact', 'koi_teq',
                'koi_insol', 'koi_model_snr', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co'
            ]
            # Make sure your df has only these columns
            X_test = df[features]
             # --- Make predictions ---
            y_pred_encoded = clf.predict(X_test)

           
            le = LabelEncoder()
            le.classes_ = np.array(["CANDIDATE", "FALSE POSITIVE", "CONFIRMED"])  # <-- convert to np.array

            y_pred_encoded = clf.predict(X_test)
            y_pred_encoded = np.array(y_pred_encoded)  # make sure predictions are np.array

            y_pred = le.inverse_transform(y_pred_encoded)

        # --- Add predictions to dataframe ---
        df["Prediction"] = y_pred

        # --- Show results ---
        st.write("### Predictions")
        st.dataframe(df)

        # --- Download results ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
        
