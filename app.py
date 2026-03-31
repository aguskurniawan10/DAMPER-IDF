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
st.set_page_config(page_title="IDF Optimizer", layout="wide")
st.title("🔥 IDF Optimizer (Stable Version)")

# =========================================================
# LOAD DATA (ANTI ERROR)
# =========================================================
url = "https://raw.githubusercontent.com/aguskurniawan10/DAMPER-IDF/main/DATA%20DISEM%20IDF%203.xlsx"

@st.cache_data
def load_data():
    try:
        df = pd.read_excel(url, sheet_name="UNIT 1")
        return df
    except Exception as e:
        st.error(f"Gagal load data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# =========================================================
# RENAME (SAFE)
# =========================================================
try:
    df.columns = [
        "time", "load", "idf_a_vane", "idf_b_vane", "fp",
        "idf_a_current", "idf_b_current", "pa_pressure",
        "fdf_a_current", "fdf_a_vane",
        "fdf_b_current", "fdf_b_vane", "airflow"
    ]
except:
    st.error("Format kolom tidak sesuai!")
    st.stop()

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

if len(df) < 10:
    st.error("Data terlalu sedikit setelah cleaning!")
    st.stop()

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
y = df["idf_a_vane"]

# =========================================================
# TRAIN MODEL (PAKAI RF → STABIL)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

st.subheader("📊 Model Performance")
st.write(f"R2: {r2:.3f}")
st.write(f"MAE: {mae:.2f}")

# =========================================================
# OPTIMIZER (ANTI ERROR)
# =========================================================
def optimize_safe(row):

    best_score = 999
    best_vane = row["idf_a_vane"]

    for vane in np.linspace(40, 90, 30):

        penalty = 0

        # SAFETY: hindari pembagian nol
        if row["idf_a_vane"] == 0:
            continue

        # simulasi current
        current_sim = row["idf_a_current"] * (vane / row["idf_a_vane"])

        # constraint furnace pressure
        if row["fp"] > -50:
            penalty += 100
        elif row["fp"] < -200:
            penalty += 50

        # current limit
        penalty += max(0, current_sim - 160) * 2

        # smooth change
        penalty += abs(vane - row["idf_a_vane"])

        if penalty < best_score:
            best_score = penalty
            best_vane = vane

    return best_vane

# =========================================================
# RUN BUTTON
# =========================================================
if st.button("🚀 Run Optimization"):

    df["idf_a_opt"] = df.apply(optimize_safe, axis=1)

    st.success("Optimization selesai!")

    # =====================================================
    # VISUAL
    # =====================================================
    st.subheader("📈 Visualization")

    fig1, ax1 = plt.subplots()
    ax1.scatter(df["idf_a_vane"], df["idf_a_opt"], alpha=0.3)
    ax1.set_title("Actual vs Optimal")
    ax1.grid()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(df["load"], df["idf_a_vane"], label="Actual", alpha=0.3)
    ax2.scatter(df["load"], df["idf_a_opt"], label="Optimal", alpha=0.3)
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

    # =====================================================
    # SUMMARY
    # =====================================================
    st.subheader("📊 Summary")

    avg_actual = df["idf_a_vane"].mean()
    avg_opt = df["idf_a_opt"].mean()

    st.metric("Actual", round(avg_actual, 2))
    st.metric("Optimal", round(avg_opt, 2))
