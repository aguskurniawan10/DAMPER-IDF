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
st.title("🔥 IDF A & B Optimization + Furnace Pressure (High Accuracy AI)")

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

# Filter lebih ketat
df = df[
    (df["load"] > 150) &
    (df["fp"] > -250) & (df["fp"] < -20) &
    (df["idf_a_current"] > 50) &
    (df["idf_b_current"] > 50)
]

st.success(f"Data siap: {df.shape}")

# =========================================================
# FEATURE ENGINEERING
# =========================================================
df["total_idf_current"] = df["idf_a_current"] + df["idf_b_current"]
df["airflow_per_load"] = df["airflow"] / df["load"]
df["vane_diff"] = df["idf_a_vane"] - df["idf_b_vane"]
df["fdf_avg"] = (df["fdf_a_vane"] + df["fdf_b_vane"]) / 2
df["air_x_pa"] = df["airflow"] * df["pa_pressure"]

features = [
    "load", "airflow", "pa_pressure",
    "idf_a_current", "idf_b_current",
    "fdf_a_vane", "fdf_b_vane",
    "total_idf_current",
    "airflow_per_load",
    "vane_diff",
    "fdf_avg",
    "air_x_pa"
]

X = df[features]
y = df["fp"]

# =========================================================
# TRAIN MODEL (RF + XGB)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# fallback jika XGBoost tidak ada
try:
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

    r2_xgb = r2_score(y_test, pred_xgb)
    mae_xgb = mean_absolute_error(y_test, pred_xgb)

except:
    xgb = None
    r2_xgb = -999
    mae_xgb = 999

# RF metrics
r2_rf = r2_score(y_test, pred_rf)
mae_rf = mean_absolute_error(y_test, pred_rf)

# pilih terbaik
if r2_xgb > r2_rf:
    model_fp = xgb
    best_name = "XGBoost"
    best_r2 = r2_xgb
    best_mae = mae_xgb
else:
    model_fp = rf
    best_name = "RandomForest"
    best_r2 = r2_rf
    best_mae = mae_rf

# =========================================================
# VALIDASI MODEL
# =========================================================
st.subheader("📊 Model Performance")

st.write(f"RF R2: {r2_rf:.3f} | MAE: {mae_rf:.2f}")
if xgb:
    st.write(f"XGB R2: {r2_xgb:.3f} | MAE: {mae_xgb:.2f}")

st.success(f"Model terbaik: {best_name}")

colA, colB = st.columns(2)
with colA:
    st.metric("R2 Final", round(best_r2, 3))
with colB:
    st.metric("MAE Final", round(best_mae, 2))

# status model
if best_r2 > 0.85:
    st.success("✅ Model sangat akurat")
elif best_r2 > 0.7:
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
st.subheader("🎛️ Input Parameter")

with st.form("form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        load = st.number_input("Load", value=None)
        airflow = st.number_input("Airflow", value=None)
        pa_pressure = st.number_input("PA Pressure", value=None)

    with col2:
        idf_a_current = st.number_input("IDF A Current", value=None)
        idf_b_current = st.number_input("IDF B Current", value=None)

    with col3:
        fdf_a_vane = st.number_input("FDF A Vane", value=None)
        fdf_b_vane = st.number_input("FDF B Vane", value=None)

    st.markdown("### IDF Existing")
    col4, col5 = st.columns(2)

    with col4:
        idf_a_vane_input = st.number_input("IDF A Vane", value=None)

    with col5:
        idf_b_vane_input = st.number_input("IDF B Vane", value=None)

    submit = st.form_submit_button("🚀 Run")

# =========================================================
# PROCESS
# =========================================================
if submit:

    if any(v is None for v in [
        load, airflow, pa_pressure,
        idf_a_current, idf_b_current,
        fdf_a_vane, fdf_b_vane,
        idf_a_vane_input, idf_b_vane_input
    ]):
        st.error("Isi semua input!")
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

    # feature tambahan
    input_data["total_idf_current"] = idf_a_current + idf_b_current
    input_data["airflow_per_load"] = airflow / load
    input_data["vane_diff"] = idf_a_vane_input - idf_b_vane_input
    input_data["fdf_avg"] = (fdf_a_vane + fdf_b_vane) / 2
    input_data["air_x_pa"] = airflow * pa_pressure

    # =====================
    # RANGE CHECK
    # =====================
    out_range = []
    for col in features:
        val = input_data[col].values[0]
        min_v, max_v = range_dict[col]
        if val < min_v or val > max_v:
            out_range.append(col)

    if out_range:
        st.warning(f"Out of range: {out_range}")

    # =====================
    # PREDIKSI
    # =====================
    pred_fp = model_fp.predict(input_data)[0]

    st.subheader("📊 Furnace Pressure")
    st.metric("Prediksi FP", round(pred_fp, 2))

    # =====================
    # CONFIDENCE
    # =====================
    confidence = max(0, 100 - (best_mae * 2))
    st.metric("Confidence (%)", round(confidence, 1))

    # =====================
    # OPTIMASI
    # =====================
    best_score = 999
    best_a = None
    best_b = None

    for a in np.linspace(40, 90, 15):
        for b in np.linspace(40, 90, 15):

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

    st.subheader("🎯 Rekomendasi")
    st.metric("IDF A", round(best_a, 2))
    st.metric("IDF B", round(best_b, 2))
