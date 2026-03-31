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
st.title("🔥 IDF Optimization + Furnace Pressure (Clean Version)")

# =========================================================
# LOAD DATA
# =========================================================
url = "https://raw.githubusercontent.com/aguskurniawan10/DAMPER-IDF/main/DATA%20DISEM%20IDF%203.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(url, sheet_name="UNIT 1")

df = load_data()

# =========================================================
# PREPROCESSING FUNCTION
# =========================================================
def preprocess(df):

    df = df.copy()

    df.columns = [
        "time", "load", "idf_a_vane", "idf_b_vane", "fp",
        "idf_a_current", "idf_b_current", "pa_pressure",
        "fdf_a_current", "fdf_a_vane",
        "fdf_b_current", "fdf_b_vane", "airflow"
    ]

    df = df.dropna()

    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.sort_values("time")

    # filter stabil
    df["load_diff"] = df["load"].diff().abs()
    df["fp_diff"] = df["fp"].diff().abs()
    df = df[(df["load_diff"] < 2) & (df["fp_diff"] < 25)]

    # remove outlier
    q_low = df["fp"].quantile(0.05)
    q_high = df["fp"].quantile(0.95)
    df = df[(df["fp"] > q_low) & (df["fp"] < q_high)]

    # smoothing
    df["fp_smooth"] = df["fp"].rolling(3).mean()

    # feature engineering
    df["total_idf_current"] = df["idf_a_current"] + df["idf_b_current"]
    df["idf_total_vane"] = df["idf_a_vane"] + df["idf_b_vane"]
    df["airflow_per_load"] = df["airflow"] / df["load"]
    df["air_per_idf"] = df["airflow"] / (df["idf_total_vane"] + 1)
    df["current_ratio"] = df["idf_a_current"] / (df["idf_b_current"] + 1)
    df["fdf_avg"] = (df["fdf_a_vane"] + df["fdf_b_vane"]) / 2
    df["air_x_pa"] = df["airflow"] * df["pa_pressure"]

    # lag
    for col in ["fp_smooth", "airflow", "idf_a_current", "idf_b_current"]:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    df = df.dropna()

    return df

df = preprocess(df)

st.success(f"Data siap: {df.shape}")

# =========================================================
# FEATURES
# =========================================================
features = [
    "load","airflow","pa_pressure",
    "idf_a_current","idf_b_current",
    "fdf_a_vane","fdf_b_vane",
    "total_idf_current","idf_total_vane",
    "airflow_per_load","air_per_idf",
    "current_ratio","fdf_avg","air_x_pa",
    "fp_smooth_lag1","fp_smooth_lag2",
    "airflow_lag1","idf_a_current_lag1","idf_b_current_lag1"
]

X = df[features]
y = df["fp_smooth"]

# =========================================================
# TRAIN MODEL
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=18,
    min_samples_split=4,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

# =========================================================
# PERFORMANCE
# =========================================================
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R2", round(r2, 3))
col2.metric("MAE", round(mae, 2))

# =========================================================
# INPUT FORM
# =========================================================
st.subheader("🎛️ Input Parameter")

with st.form("form"):

    load = st.number_input("Load", value=300.0)
    airflow = st.number_input("Airflow", value=900.0)
    pa_pressure = st.number_input("PA Pressure", value=7.0)

    idf_a_current = st.number_input("IDF A Current", value=200.0)
    idf_b_current = st.number_input("IDF B Current", value=200.0)

    fdf_a_vane = st.number_input("FDF A Vane", value=45.0)
    fdf_b_vane = st.number_input("FDF B Vane", value=45.0)

    idf_a_vane_input = st.number_input("IDF A Vane Existing", value=95.0)
    idf_b_vane_input = st.number_input("IDF B Vane Existing", value=95.0)

    submit = st.form_submit_button("🚀 Run")

# =========================================================
# FUNCTION: BUILD FEATURE INPUT
# =========================================================
def build_input():

    data = pd.DataFrame([{
        "load": load,
        "airflow": airflow,
        "pa_pressure": pa_pressure,
        "idf_a_current": idf_a_current,
        "idf_b_current": idf_b_current,
        "fdf_a_vane": fdf_a_vane,
        "fdf_b_vane": fdf_b_vane
    }])

    data["total_idf_current"] = idf_a_current + idf_b_current
    data["idf_total_vane"] = idf_a_vane_input + idf_b_vane_input
    data["airflow_per_load"] = airflow / load
    data["air_per_idf"] = airflow / (data["idf_total_vane"] + 1)
    data["current_ratio"] = idf_a_current / (idf_b_current + 1)
    data["fdf_avg"] = (fdf_a_vane + fdf_b_vane) / 2
    data["air_x_pa"] = airflow * pa_pressure

    data["fp_smooth_lag1"] = -100
    data["fp_smooth_lag2"] = -100
    data["airflow_lag1"] = airflow
    data["idf_a_current_lag1"] = idf_a_current
    data["idf_b_current_lag1"] = idf_b_current

    return data

# =========================================================
# PROCESS
# =========================================================
if submit:

    st.write("✅ Submit berhasil → model jalan")

    input_data = build_input()

    # =====================
    # PREDIKSI
    # =====================
    pred_fp = model.predict(input_data)[0]

    st.subheader("📊 Furnace Pressure Prediction")
    st.metric("FP (Predicted)", round(pred_fp, 2))

    # =====================
    # OPTIMIZER (PASTI JALAN)
    # =====================
    st.subheader("🎯 Rekomendasi Damper")

    target_fp = -100

    best_score = 999
    best_a, best_b, best_fp = None, None, None

    for a in np.linspace(40, 95, 15):
        for b in np.linspace(40, 95, 15):

            temp = build_input()

            temp["idf_a_current"] = idf_a_current * (a / idf_a_vane_input)
            temp["idf_b_current"] = idf_b_current * (b / idf_b_vane_input)

            temp["total_idf_current"] = temp["idf_a_current"] + temp["idf_b_current"]
            temp["idf_total_vane"] = a + b
            temp["air_per_idf"] = airflow / (temp["idf_total_vane"] + 1)
            temp["current_ratio"] = temp["idf_a_current"] / (temp["idf_b_current"] + 1)

            temp["fp_smooth_lag1"] = pred_fp
            temp["fp_smooth_lag2"] = pred_fp

            pred_sim = model.predict(temp)[0]

            penalty = abs(pred_sim - target_fp) * 3

            if pred_sim > -50:
                penalty += 1000

            penalty += abs(a - b) * 0.3

            if penalty < best_score:
                best_score = penalty
                best_a = a
                best_b = b
                best_fp = pred_sim

    colA, colB, colC = st.columns(3)

    colA.metric("IDF A Optimal", round(best_a, 2))
    colB.metric("IDF B Optimal", round(best_b, 2))
    colC.metric("FP Setelah Optimasi", round(best_fp, 2))
