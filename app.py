import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="IDF Optimizer AI", layout="wide")
st.title("🔥 IDF Optimization (Dual Model - Physics Aware)")

# =========================================================
# LOAD DATA
# =========================================================
url = "https://raw.githubusercontent.com/aguskurniawan10/DAMPER-IDF/main/DATA%20DISEM%20IDF%203.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(url, sheet_name="UNIT 1")

df = load_data()

# =========================================================
# PREPROCESS
# =========================================================
def preprocess(df):

    df = df.copy()

    df.columns = [
        "time","load","idf_a_vane","idf_b_vane","fp",
        "idf_a_current","idf_b_current","pa_pressure",
        "fdf_a_current","fdf_a_vane",
        "fdf_b_current","fdf_b_vane","airflow"
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

    return df

df = preprocess(df)

st.success(f"Data siap: {df.shape}")

# =========================================================
# 🔥 MODEL 1: DAMPER → AIRFLOW
# =========================================================
df["idf_total_vane"] = df["idf_a_vane"] + df["idf_b_vane"]

features_air = [
    "idf_total_vane",
    "load",
    "pa_pressure"
]

X_air = df[features_air]
y_air = df["airflow"]

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_air, y_air, test_size=0.2, random_state=42
)

model_air = RandomForestRegressor(n_estimators=200, max_depth=12)
model_air.fit(X_train_a, y_train_a)

pred_air = model_air.predict(X_test_a)

r2_air = r2_score(y_test_a, pred_air)

# =========================================================
# 🔥 MODEL 2: AIRFLOW → FP
# =========================================================
features_fp = [
    "airflow",
    "load",
    "idf_a_current",
    "idf_b_current",
    "pa_pressure",
    "fdf_a_vane",
    "fdf_b_vane"
]

X_fp = df[features_fp]
y_fp = df["fp"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fp, y_fp, test_size=0.2, random_state=42
)

model_fp = RandomForestRegressor(n_estimators=300, max_depth=15)
model_fp.fit(X_train_f, y_train_f)

pred_fp = model_fp.predict(X_test_f)

r2_fp = r2_score(y_test_f, pred_fp)

# =========================================================
# PERFORMANCE
# =========================================================
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("R2 Airflow Model", round(r2_air, 3))
col2.metric("R2 FP Model", round(r2_fp, 3))

# =========================================================
# INPUT
# =========================================================
st.subheader("🎛️ Input Parameter")

with st.form("form"):

    load = st.number_input("Load", value=300.0)
    pa_pressure = st.number_input("PA Pressure", value=7.0)

    idf_a_current = st.number_input("IDF A Current", value=200.0)
    idf_b_current = st.number_input("IDF B Current", value=200.0)

    fdf_a_vane = st.number_input("FDF A Vane", value=45.0)
    fdf_b_vane = st.number_input("FDF B Vane", value=45.0)

    idf_a_vane_input = st.number_input("IDF A Vane", value=95.0)
    idf_b_vane_input = st.number_input("IDF B Vane", value=95.0)

    submit = st.form_submit_button("🚀 Run")

# =========================================================
# PROCESS
# =========================================================
if submit:

    # =====================
    # CURRENT AIRFLOW
    # =====================
    total_vane_now = idf_a_vane_input + idf_b_vane_input

    airflow_now = model_air.predict(pd.DataFrame([{
        "idf_total_vane": total_vane_now,
        "load": load,
        "pa_pressure": pa_pressure
    }]))[0]

    # =====================
    # CURRENT FP
    # =====================
    fp_now = model_fp.predict(pd.DataFrame([{
        "airflow": airflow_now,
        "load": load,
        "idf_a_current": idf_a_current,
        "idf_b_current": idf_b_current,
        "pa_pressure": pa_pressure,
        "fdf_a_vane": fdf_a_vane,
        "fdf_b_vane": fdf_b_vane
    }]))[0]

    st.subheader("📊 Kondisi Saat Ini")
    st.metric("Airflow", round(airflow_now, 2))
    st.metric("Furnace Pressure", round(fp_now, 2))

    # =====================================================
    # 🔥 OPTIMIZER (REAL WORKING)
    # =====================================================
    st.subheader("🎯 Rekomendasi Damper")

    target_fp = -100

    best_score = 999
    best_total = None
    best_fp = None

    for total in np.linspace(80, 180, 50):

        # airflow dari model
        airflow_sim = model_air.predict(pd.DataFrame([{
            "idf_total_vane": total,
            "load": load,
            "pa_pressure": pa_pressure
        }]))[0]

        # split damper
        a = total / 2
        b = total / 2

        # current scaling
        idf_a_cur = idf_a_current * (a / idf_a_vane_input)
        idf_b_cur = idf_b_current * (b / idf_b_vane_input)

        # FP prediction
        pred_sim = model_fp.predict(pd.DataFrame([{
            "airflow": airflow_sim,
            "load": load,
            "idf_a_current": idf_a_cur,
            "idf_b_current": idf_b_cur,
            "pa_pressure": pa_pressure,
            "fdf_a_vane": fdf_a_vane,
            "fdf_b_vane": fdf_b_vane
        }]))[0]

        penalty = abs(pred_sim - target_fp)

        if pred_sim > -50:
            penalty += 1000

        if penalty < best_score:
            best_score = penalty
            best_total = total
            best_fp = pred_sim

    best_a = best_total / 2
    best_b = best_total / 2

    colA, colB, colC = st.columns(3)

    colA.metric("IDF A Optimal", round(best_a, 2))
    colB.metric("IDF B Optimal", round(best_b, 2))
    colC.metric("FP Setelah Optimasi", round(best_fp, 2))
