import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="IDF Optimizer AI", layout="wide")
st.title("🔥 IDF Optimization + Furnace Pressure (Time-Series AI)")

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
    (df["fp"] > -250) & (df["fp"] < -20)
]

# =========================================================
# SORT TIME
# =========================================================
df = df.sort_values("time")

# =========================================================
# FEATURE ENGINEERING
# =========================================================
df["total_idf_current"] = df["idf_a_current"] + df["idf_b_current"]
df["airflow_per_load"] = df["airflow"] / df["load"]
df["vane_diff"] = df["idf_a_vane"] - df["idf_b_vane"]
df["fdf_avg"] = (df["fdf_a_vane"] + df["fdf_b_vane"]) / 2
df["air_x_pa"] = df["airflow"] * df["pa_pressure"]

# =========================================================
# LAG FEATURE (WAJIB)
# =========================================================
lag_cols = ["fp", "airflow", "idf_a_current", "idf_b_current", "pa_pressure"]

for col in lag_cols:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag2"] = df[col].shift(2)

df = df.dropna()

st.success(f"Data siap: {df.shape}")

# =========================================================
# FEATURE LIST
# =========================================================
features = [
    "load", "airflow", "pa_pressure",
    "idf_a_current", "idf_b_current",
    "fdf_a_vane", "fdf_b_vane",

    "total_idf_current",
    "airflow_per_load",
    "vane_diff",
    "fdf_avg",
    "air_x_pa",

    "fp_lag1", "fp_lag2",
    "airflow_lag1", "idf_a_current_lag1",
    "idf_b_current_lag1", "pa_pressure_lag1"
]

X = df[features]
y = df["fp"]

# =========================================================
# TRAIN MODEL
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=300, max_depth=15)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

r2 = r2_score(y_test, pred_rf)
mae = mean_absolute_error(y_test, pred_rf)

st.subheader("📊 Model Performance")
st.metric("R2", round(r2, 3))
st.metric("MAE", round(mae, 2))

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

    # lag (sementara pakai kondisi sekarang)
    input_data["fp_lag1"] = -100
    input_data["fp_lag2"] = -100
    input_data["airflow_lag1"] = airflow
    input_data["idf_a_current_lag1"] = idf_a_current
    input_data["idf_b_current_lag1"] = idf_b_current
    input_data["pa_pressure_lag1"] = pa_pressure

    # =====================
    # PREDIKSI
    # =====================
    pred_fp = rf.predict(input_data)[0]

    st.subheader("📊 Furnace Pressure")
    st.metric("Prediksi FP", round(pred_fp, 2))

    # =====================
    # OPTIMASI
    # =====================
    best_score = 999
    best_a, best_b = None, None

    for a in np.linspace(40, 90, 15):
        for b in np.linspace(40, 90, 15):

            temp = input_data.copy()

            temp["idf_a_current"] = idf_a_current * (a / idf_a_vane_input)
            temp["idf_b_current"] = idf_b_current * (b / idf_b_vane_input)

            pred_sim = rf.predict(temp)[0]

            penalty = abs(pred_sim + 100) * 2
            penalty += abs(a - b) * 0.5

            if pred_sim > -50:
                penalty += 300

            if penalty < best_score:
                best_score = penalty
                best_a, best_b = a, b

    st.subheader("🎯 Rekomendasi")
    st.metric("IDF A", round(best_a, 2))
    st.metric("IDF B", round(best_b, 2))
