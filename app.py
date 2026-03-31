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
st.title("🔥 IDF Damper Optimization + Furnace Pressure Prediction")

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

# =========================================================
# MODEL 1: IDF
# =========================================================
y_idf = df["idf_a_vane"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_idf, test_size=0.2, random_state=42
)

model_idf = RandomForestRegressor(n_estimators=100, max_depth=10)
model_idf.fit(X_train, y_train)

pred_idf = model_idf.predict(X_test)

r2_idf = r2_score(y_test, pred_idf)
mae_idf = mean_absolute_error(y_test, pred_idf)

# =========================================================
# MODEL 2: FURNACE PRESSURE
# =========================================================
y_fp = df["fp"]

X_train_fp, X_test_fp, y_train_fp, y_test_fp = train_test_split(
    X, y_fp, test_size=0.2, random_state=42
)

model_fp = RandomForestRegressor(n_estimators=100, max_depth=10)
model_fp.fit(X_train_fp, y_train_fp)

pred_fp = model_fp.predict(X_test_fp)

r2_fp = r2_score(y_test_fp, pred_fp)
mae_fp = mean_absolute_error(y_test_fp, pred_fp)

# =========================================================
# PERFORMANCE
# =========================================================
st.subheader("📊 Model Performance")

perf_df = pd.DataFrame({
    "Model": ["IDF Model", "Furnace Pressure Model"],
    "R2": [r2_idf, r2_fp],
    "MAE": [mae_idf, mae_fp]
})

st.dataframe(perf_df)

# =========================================================
# INPUT USER
# =========================================================
st.subheader("🎛️ Input Parameter Operasi")

col1, col2, col3 = st.columns(3)

with col1:
    load = st.number_input("Load (MW)", value=200.0)
    airflow = st.number_input("Airflow", value=750.0)
    pa_pressure = st.number_input("PA Pressure", value=8.5)

with col2:
    idf_a_current = st.number_input("IDF A Current", value=150.0)
    idf_b_current = st.number_input("IDF B Current", value=150.0)

with col3:
    fdf_a_vane = st.number_input("FDF A Vane (%)", value=40.0)
    fdf_b_vane = st.number_input("FDF B Vane (%)", value=40.0)

# =========================================================
# PREDIKSI
# =========================================================
if st.button("🔍 Prediksi Furnace Pressure"):

    input_data = pd.DataFrame([{
        "load": load,
        "airflow": airflow,
        "pa_pressure": pa_pressure,
        "idf_a_current": idf_a_current,
        "idf_b_current": idf_b_current,
        "fdf_a_vane": fdf_a_vane,
        "fdf_b_vane": fdf_b_vane
    }])

    pred_fp = model_fp.predict(input_data)[0]

    st.subheader("📊 Hasil Prediksi")
    st.metric("Furnace Pressure", round(pred_fp, 2))

    if pred_fp > -50:
        st.error("⚠️ Furnace Pressure terlalu positif")
    elif pred_fp < -200:
        st.warning("⚠️ Furnace Pressure terlalu negatif")
    else:
        st.success("✅ Furnace Pressure normal")

# =========================================================
# OPTIMASI
# =========================================================
if st.button("🚀 Cari IDF Optimal"):

    input_data = pd.DataFrame([{
        "load": load,
        "airflow": airflow,
        "pa_pressure": pa_pressure,
        "idf_a_current": idf_a_current,
        "idf_b_current": idf_b_current,
        "fdf_a_vane": fdf_a_vane,
        "fdf_b_vane": fdf_b_vane
    }])

    best_score = 999
    best_vane = None

    for vane in np.linspace(40, 90, 40):

        temp = input_data.copy()

        if idf_a_current != 0:
            temp["idf_a_current"] = idf_a_current * (vane / 70)

        pred_fp = model_fp.predict(temp)[0]

        penalty = 0
        target_fp = -100

        penalty += abs(pred_fp - target_fp) * 2

        if pred_fp > -50:
            penalty += 300

        penalty += max(0, temp["idf_a_current"].values[0] - 160) * 3

        if penalty < best_score:
            best_score = penalty
            best_vane = vane

    st.subheader("🎯 Rekomendasi")
    st.metric("IDF A Optimal (%)", round(best_vane, 2))

# =========================================================
# VISUAL DATA HISTORIS
# =========================================================
st.subheader("📈 Data Historis")

fig, ax = plt.subplots()
ax.scatter(df["idf_a_vane"], df["fp"], alpha=0.3)
ax.set_xlabel("IDF A Vane")
ax.set_ylabel("Furnace Pressure")
ax.set_title("Historis: Damper vs Furnace Pressure")
ax.grid()

st.pyplot(fig)
