import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.set_page_config(page_title="IDF Optimizer", layout="wide")

st.title("🔥 IDF Damper Optimization (Auto Load GitHub)")

# =========================================================
# LOAD DATA DARI GITHUB
# =========================================================
url = "https://raw.githubusercontent.com/aguskurniawan10/DAMPER-IDF/main/DATA%20DISEM%20IDF%203.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(url, sheet_name="UNIT 1")

df = load_data()

st.success("Data berhasil di-load dari GitHub ✅")

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

st.write("Jumlah data:", df.shape)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# MODEL
# =========================================================
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=6)
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    results.append([name, r2, mae])
    trained_models[name] = model

perf_df = pd.DataFrame(results, columns=["Model", "R2", "MAE"])
st.dataframe(perf_df)

best_model_name = perf_df.sort_values("R2", ascending=False).iloc[0]["Model"]
best_model = trained_models[best_model_name]

st.success(f"Model terbaik: {best_model_name}")

# =========================================================
# OPTIMIZER
# =========================================================
def optimize(row):
    best_score = 999
    best_vane = None

    for vane in np.linspace(40, 90, 50):

        penalty = 0

        if row["fp"] > -50:
            penalty += 100
        elif row["fp"] < -200:
            penalty += 50

        penalty += max(0, row["idf_a_current"] - 160) * 2

        score = abs(vane - row["idf_a_vane"]) + penalty

        if score < best_score:
            best_score = score
            best_vane = vane

    return best_vane

df["idf_a_opt"] = df.apply(optimize, axis=1)

# =========================================================
# VISUAL
# =========================================================
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
ax2.set_title("Load vs Damper")
ax2.grid()
st.pyplot(fig2)

# =========================================================
# SUMMARY
# =========================================================
avg_actual = df["idf_a_vane"].mean()
avg_opt = df["idf_a_opt"].mean()
saving = avg_actual - avg_opt

st.subheader("📊 Summary")
st.metric("Actual", round(avg_actual, 2))
st.metric("Optimal", round(avg_opt, 2))
st.metric("Saving", round(saving, 2))
