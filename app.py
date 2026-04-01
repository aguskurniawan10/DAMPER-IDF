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
    df["load_diff"] = df["load"].diff().abs()
    df["fp_diff"]   = df["fp"].diff().abs()
    df = df[(df["load_diff"] < 2) & (df["fp_diff"] < 25)]
    return df

df = preprocess(df)
st.success(f"Data siap: {df.shape[0]} baris")

# =========================================================
# MODEL 1: DAMPER → AIRFLOW
# =========================================================
df["idf_total_vane"] = df["idf_a_vane"] + df["idf_b_vane"]

features_air = ["idf_total_vane", "load", "pa_pressure"]
X_air = df[features_air]
y_air = df["airflow"]

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_air, y_air, test_size=0.2, random_state=42
)
model_air = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
model_air.fit(X_train_a, y_train_a)
r2_air = r2_score(y_test_a, model_air.predict(X_test_a))

# =========================================================
# MODEL 2: AIRFLOW → FP
# =========================================================
features_fp = [
    "airflow", "load",
    "idf_a_current", "idf_b_current",
    "pa_pressure", "fdf_a_vane", "fdf_b_vane"
]
X_fp = df[features_fp]
y_fp = df["fp"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fp, y_fp, test_size=0.2, random_state=42
)
model_fp = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
model_fp.fit(X_train_f, y_train_f)
r2_fp = r2_score(y_test_f, model_fp.predict(X_test_f))

# =========================================================
# PERFORMANCE
# =========================================================
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R² Airflow Model", round(r2_air, 4))
col2.metric("R² FP Model",      round(r2_fp,  4))

# =========================================================
# INPUT
# =========================================================
st.subheader("🎛️ Input Parameter Operasi")

with st.form("form_input"):
    col_l, col_p = st.columns(2)
    load        = col_l.number_input("Load (MW)",          value=300.0, step=1.0)
    pa_pressure = col_p.number_input("PA Pressure (KPA)", value=7.0,   step=0.1)

    st.markdown("**IDF Parameter**")
    c1, c2, c3, c4 = st.columns(4)
    idf_a_current  = c1.number_input("IDF A Current (A)",       value=200.0, step=1.0)
    idf_b_current  = c2.number_input("IDF B Current (A)",       value=200.0, step=1.0)
    idf_a_vane_now = c3.number_input("IDF A Vane saat ini (%)", value=95.0,  step=1.0)
    idf_b_vane_now = c4.number_input("IDF B Vane saat ini (%)", value=95.0,  step=1.0)

    st.markdown("**FDF Parameter**")
    d1, d2 = st.columns(2)
    fdf_a_vane = d1.number_input("FDF A Vane (%)", value=45.0, step=1.0)
    fdf_b_vane = d2.number_input("FDF B Vane (%)", value=45.0, step=1.0)

    submit = st.form_submit_button("🚀 Hitung & Optimalkan")

# =========================================================
# PROCESS
# =========================================================
if submit:

    # ---------- Kondisi saat ini ----------
    total_vane_now = idf_a_vane_now + idf_b_vane_now

    airflow_now = model_air.predict(pd.DataFrame([{
        "idf_total_vane": total_vane_now,
        "load":           load,
        "pa_pressure":    pa_pressure
    }]))[0]

    fp_now = model_fp.predict(pd.DataFrame([{
        "airflow":       airflow_now,
        "load":          load,
        "idf_a_current": idf_a_current,
        "idf_b_current": idf_b_current,
        "pa_pressure":   pa_pressure,
        "fdf_a_vane":    fdf_a_vane,
        "fdf_b_vane":    fdf_b_vane
    }]))[0]

    st.subheader("📌 Kondisi Saat Ini")
    m1, m2, m3 = st.columns(3)
    m1.metric("IDF A Vane",               f"{idf_a_vane_now:.1f} %")
    m2.metric("IDF B Vane",               f"{idf_b_vane_now:.1f} %")
    m3.metric("Furnace Pressure (prediksi)", f"{fp_now:.1f} Pa")

    # ---------- Optimizer: 2D grid search ----------
    st.subheader("🎯 Rekomendasi Bukaan Damper IDF")
    st.caption("Target Furnace Pressure: **-100 s/d -150 Pa**")

    FP_LOW  = -150   # batas bawah
    FP_HIGH = -100   # batas atas
    FP_MID  = (FP_LOW + FP_HIGH) / 2   # -125, titik tengah target

    vane_range = np.arange(40, 101, 2)  # 40% – 100%, step 2%

    results = []

    for a in vane_range:
        for b in vane_range:
            total = a + b

            airflow_sim = model_air.predict(pd.DataFrame([{
                "idf_total_vane": total,
                "load":           load,
                "pa_pressure":    pa_pressure
            }]))[0]

            fp_sim = model_fp.predict(pd.DataFrame([{
                "airflow":       airflow_sim,
                "load":          load,
                "idf_a_current": idf_a_current,
                "idf_b_current": idf_b_current,
                "pa_pressure":   pa_pressure,
                "fdf_a_vane":    fdf_a_vane,
                "fdf_b_vane":    fdf_b_vane
            }]))[0]

            # Hanya simpan jika FP dalam rentang target
            if FP_LOW <= fp_sim <= FP_HIGH:
                # Skor: prioritas FP mendekati tengah target, sekunder total bukaan kecil
                score = abs(fp_sim - FP_MID) + 0.05 * (a + b)
                results.append({
                    "IDF A Vane (%)":      a,
                    "IDF B Vane (%)":      b,
                    "Prediksi FP (Pa)":  round(fp_sim, 1),
                    "_score":              score
                })

    if results:
        res_df = (
            pd.DataFrame(results)
            .sort_values("_score")
            .drop(columns=["_score"])
            .reset_index(drop=True)
        )

        top_df = res_df.head(10).copy()
        top_df.index = top_df.index + 1
        top_df.index.name = "Rank"

        st.dataframe(
            top_df.style
                .format({
                    "IDF A Vane (%)":     "{:.0f}",
                    "IDF B Vane (%)":     "{:.0f}",
                    "Prediksi FP (mmWC)": "{:.1f}"
                })
                .background_gradient(subset=["Prediksi FP (Pa)"], cmap="RdYlGn_r"),
            use_container_width=True
        )

        best = res_df.iloc[0]
        st.success(
            f"✅ **Rekomendasi Terbaik** — "
            f"IDF A Vane: **{best['IDF A Vane (%)']:.0f}%** | "
            f"IDF B Vane: **{best['IDF B Vane (%)']:.0f}%** | "
            f"Prediksi FP: **{best['Prediksi FP (Pa)']:.1f} Pa**"
        )

        st.info(
            f"Ditemukan **{len(res_df)}** kombinasi damper yang memenuhi target FP -100 s/d -150 Pa. "
            f"Tabel menampilkan **Top 10** — diurutkan berdasarkan kedekatan FP ke titik tengah target (-125 Pa)."
        )

    else:
        st.warning(
            "⚠️ Tidak ditemukan kombinasi damper yang menghasilkan FP dalam rentang -100 s/d -150 Pa "
            "pada kondisi operasi ini. Coba sesuaikan nilai FDF Vane atau PA Pressure."
        )
