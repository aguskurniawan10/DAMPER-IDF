import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="IDF Optimizer AI", layout="wide")
st.title("🔥 IDF A & B Optimization + Furnace Pressure Prediction (Validated AI)")

# =========================================================
# LOAD DATA
# =========================================================
url = "https://raw.githubusercontent.com/aguskurniawan10/DAMPER-IDF/main/DATA%20DISEM%20IDF%203.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(url, sheet_name="UNIT 1")

df = load_data()

# =========================================================
# RENAME
# =========================================================
df.columns = [
    "time", "load", "idf_a_vane", "idf_b_vane", "fp",
    "idf_a_current", "idf_b_current", "pa_pressure",
    "fdf_a_current", "fdf_a_vane",
    "fdf_b_current", "fdf_b_vane", "airflow"
]

# =========================================================
# CLEANING
# =========================================================
df = df.dropna()

for col in df.columns:
    if col != "time":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

df = df[
    (df["load"] > 150) &
    (df["fp"] > -300) &
    (df["fp"] < 50)
]

st.success(f"Data siap: {df.shape}")

# =========================================================
# FEATURE
# =========================================================
features = [
    "load", "airflow", "pa_pressure",
    "idf_a_current", "idf_b_current",
    "fdf_a_vane", "fdf_b_vane"
]

X = df[features]
y_fp = df["fp"]

# =========================================================
# TRAIN MODEL
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_fp, test_size=0.2, random_state=42
)

model_fp = RandomForestRegressor(n_estimators=100, max_depth=10)
model_fp.fit(X_train, y_train)

# =========================================================
# VALIDASI MODEL
# =========================================================
st.subheader("📊 Validasi Model")

pred_test = model_fp.predict(X_test)

r2 = r2_score(y_test, pred_test)
mae = mean_absolute_error(y_test, pred_test)

colA, colB = st.columns(2)

with colA:
    st.metric("R2 Score", round(r2, 3))
with colB:
    st.metric("MAE", round(mae, 2))

if r2 > 0.85:
    st.success("✅ Model sangat baik (reliable)")
elif r2 > 0.7:
    st.warning("⚠️ Model cukup baik")
else:
    st.error("❌ Model kurang akurat")

# =========================================================
# RANGE TRAINING
# =========================================================
range_dict = {col: (df[col].min(), df[col].max()) for col in features}

# =========================================================
# INPUT FORM
# =========================================================
st.subheader("🎛️ Input Parameter Operasi")

with st.form("form_input"):

    col1, col2, col3 = st.columns(3)

    with col1:
        load = st.number_input("Load (MW)", value=None)
        airflow = st.number_input("Airflow", value=None)
        pa_pressure = st.number_input("PA Pressure", value=None)

    with col2:
        idf_a_current = st.number_input("IDF A Current", value=None)
        idf_b_current = st.number_input("IDF B Current", value=None)

    with col3:
        fdf_a_vane = st.number_input("FDF A Vane (%)", value=None)
        fdf_b_vane = st.number_input("FDF B Vane (%)", value=None)

    st.markdown("### 🔧 IDF Existing")
    col4, col5 = st.columns(2)

    with col4:
        idf_a_vane_input = st.number_input("IDF A Vane (%)", value=None)

    with col5:
        idf_b_vane_input = st.number_input("IDF B Vane (%)", value=None)

    submit = st.form_submit_button("🚀 Jalankan Prediksi & Optimasi")

# =========================================================
# PROCESS
# =========================================================
if submit:

    inputs = [
        load, airflow, pa_pressure,
        idf_a_current, idf_b_current,
        fdf_a_vane, fdf_b_vane,
        idf_a_vane_input, idf_b_vane_input
    ]

    if any(v is None for v in inputs):
        st.error("⚠️ Semua input harus diisi!")
        st.stop()

    input_data = pd.DataFrame([{
        "load": load,
        "airflow": airflow,
        "pa_pressure": pa_pressure,
        "idf_a_current": idf_a_current,
        "idf_b_current": idf_b_current,
        "fdf_a_vane": fdf_a_vane,
        "fdf_b_vane": fdf_b_vane
    }])

    # =====================================================
    # VALIDASI RANGE INPUT
    # =====================================================
    out_of_range = []

    for col in features:
        val = input_data[col].values[0]
        min_val, max_val = range_dict[col]

        if val < min_val or val > max_val:
            out_of_range.append(col)

    if out_of_range:
        st.warning(f"⚠️ Input di luar range training: {out_of_range}")

    # =====================================================
    # PREDIKSI FP
    # =====================================================
    pred_fp = model_fp.predict(input_data)[0]

    st.subheader("📊 Hasil Prediksi")
    st.metric("Furnace Pressure", round(pred_fp, 2))

    if pred_fp > -50:
        st.error("⚠️ Pressure terlalu positif")
    elif pred_fp < -200:
        st.warning("⚠️ Pressure terlalu negatif")
    else:
        st.success("✅ Normal")

    # =====================================================
    # CONFIDENCE
    # =====================================================
    st.subheader("🧠 Confidence")

    confidence = max(0, 100 - (mae * 2))
    st.metric("Confidence (%)", round(confidence, 1))

    # =====================================================
    # OPTIMASI IDF A & B
    # =====================================================
    best_score = 999
    best_a = None
    best_b = None

    for a in np.linspace(40, 90, 20):
        for b in np.linspace(40, 90, 20):

            temp = input_data.copy()

            if idf_a_vane_input != 0:
                temp["idf_a_current"] = idf_a_current * (a / idf_a_vane_input)

            if idf_b_vane_input != 0:
                temp["idf_b_current"] = idf_b_current * (b / idf_b_vane_input)

            pred_sim = model_fp.predict(temp)[0]

            penalty = abs(pred_sim + 100) * 2

            if pred_sim > -50:
                penalty += 300

            penalty += abs(a - b) * 0.5

            if penalty < best_score:
                best_score = penalty
                best_a = a
                best_b = b

    st.subheader("🎯 Rekomendasi Optimal")

    colA, colB = st.columns(2)

    with colA:
        st.metric("IDF A (%)", round(best_a, 2))

    with colB:
        st.metric("IDF B (%)", round(best_b, 2))
