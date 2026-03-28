import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Predictor · ML Pipeline",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ─ Main background ─ */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e8e8f0;
    }

    /* ─ Sidebar ─ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid rgba(100, 120, 255, 0.2);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa !important;
    }

    /* ─ Metric cards ─ */
    [data-testid="metric-container"] {
        background: rgba(100, 120, 255, 0.08);
        border: 1px solid rgba(100, 120, 255, 0.25);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    [data-testid="metric-container"] label {
        color: #a78bfa !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* ─ Buttons ─ */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
    }

    /* ─ Inputs & selects ─ */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(100, 120, 255, 0.3) !important;
        border-radius: 8px !important;
        color: #e8e8f0 !important;
    }

    /* ─ Section headings ─ */
    h1 { color: #a78bfa !important; font-weight: 700; }
    h2 { color: #c4b5fd !important; font-weight: 600; }
    h3 { color: #ddd6fe !important; font-weight: 500; }

    /* ─ Result boxes ─ */
    .result-high {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-sub {
        font-size: 1.1rem;
        opacity: 0.8;
    }

    /* ─ Cards ─ */
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(100, 120, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ─ Tab styling ─ */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a78bfa;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }

    /* ─ DataFrame ─ */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ─ Progress bar ─ */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        border-radius: 99px;
    }

    /* ─ Divider ─ */
    hr { border-color: rgba(100, 120, 255, 0.2) !important; }

    /* ─ Expander ─ */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 8px !important;
        color: #c4b5fd !important;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
]
EDUCATION_OPTIONS = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
    "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
    "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"
]
MARITAL_OPTIONS = [
    "Never-married", "Married-civ-spouse", "Divorced",
    "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"
]
OCCUPATION_OPTIONS = [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
]
RELATIONSHIP_OPTIONS = [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
]
RACE_OPTIONS = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
]
SEX_OPTIONS = ["Male", "Female"]
COUNTRY_OPTIONS = [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
    "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
    "South", "China", "Cuba", "Iran", "Honduras", "Philippines",
    "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
    "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
    "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
    "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
    "Peru", "Hong", "Holand-Netherlands"
]

EDUCATION_NUM_MAP = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13,
    "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

ARTIFACTS_DIR = "artifacts"
EDA_DIR = os.path.join(ARTIFACTS_DIR, "eda_plots")


def normalize_income_col(df: pd.DataFrame) -> pd.DataFrame:
    """Safely normalize the income column regardless of whether it's str or int."""
    df = df.copy()
    if "income" not in df.columns:
        return df
    if df["income"].dtype == object:
        df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
    else:
        # Already numeric (0/1) — map back to readable labels
        df["income"] = df["income"].map({0: "<=50K", 1: ">50K"}).fillna(df["income"].astype(str))
    return df


# ─── Helper functions ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
        prep_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(prep_path, "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor, True
    except Exception as e:
        return None, None, False


@st.cache_data
def load_model_report():
    path = os.path.join(ARTIFACTS_DIR, "model_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_raw_data():
    path = os.path.join(ARTIFACTS_DIR, "raw.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["net_capital"] = df["capital_gain"] - df["capital_loss"]
    df["work_intensity"] = df["hours_per_week"] * df["age"]
    df["high_education"] = (df["education_num"] >= 13).astype(int)
    df["is_married"] = df["marital_status"].isin(
        ["Married-civ-spouse", "Married-AF-spouse"]
    ).astype(int)
    df["has_capital"] = ((df["capital_gain"] > 0) | (df["capital_loss"] > 0)).astype(int)
    return df


def make_prediction(model, preprocessor, input_data: dict):
    df = pd.DataFrame([input_data])
    df = engineer_features(df)
    arr = preprocessor.transform(df)
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0] if hasattr(model, "predict_proba") else None
    return pred, prob


def render_gauge(probability: float):
    """Render a circular gauge showing income probability."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    theta_start = np.pi
    theta_end = 0
    theta_fill = theta_start - probability * np.pi

    # Background arc
    theta_bg = np.linspace(theta_end, theta_start, 100)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), linewidth=20,
            color="#2d2d4e", solid_capstyle="round")
    # Filled arc
    theta_fill_arr = np.linspace(theta_fill, theta_start, 100)
    color = "#10b981" if probability > 0.5 else "#ef4444"
    ax.plot(np.cos(theta_fill_arr), np.sin(theta_fill_arr), linewidth=20,
            color=color, solid_capstyle="round")
    # Text
    ax.text(0, -0.2, f"{probability*100:.1f}%", ha="center", va="center",
            fontsize=28, fontweight="bold", color="white",
            fontfamily="monospace")
    ax.text(0, -0.55, "Probability >50K", ha="center", va="center",
            fontsize=9, color="#a78bfa")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.8, 1.3)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Income Predictor")
    st.markdown("*ML Pipeline · Production Ready*")
    st.divider()

    model, preprocessor, model_ready = load_model_and_preprocessor()

    if model_ready:
        st.success("✅ Model loaded")
        st.markdown(f"**Model type:** `{type(model).__name__}`")
    else:
        st.warning("⚠️ Model not trained yet")
        st.markdown("Go to **🚂 Train Model** tab to train.")

    st.divider()

    # Quick training trigger
    st.markdown("### 🚀 Quick Actions")
    if st.button("🔄 Retrain Model", use_container_width=True):
        st.session_state["trigger_training"] = True

    st.divider()
    st.markdown("""
    **Dataset:** Adult Census Income  
    **Source:** UCI ML Repository  
    **Task:** Binary Classification  
    **Target:** Income ≤50K vs >50K
    """)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
    <h1 style="font-size:3rem; margin:0;">💰 Income Prediction</h1>
    <p style="color:#a78bfa; font-size:1.2rem; margin-top:0.5rem;">
        Production-Grade ML Pipeline · Adult Census Income Dataset
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🎯 Predict", "📊 EDA", "🏆 Model Performance",
    "🚂 Train Model", "📋 Data Explorer"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 · PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Enter Individual Details")
    st.markdown("Fill in the form below and click **Predict Income** to get a result.")

    with st.form("prediction_form"):
        # ── Row 1 ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("🎂 Age", min_value=17, max_value=90, value=35, step=1)
        with c2:
            sex = st.selectbox("⚧ Gender", SEX_OPTIONS)
        with c3:
            race = st.selectbox("🌍 Race", RACE_OPTIONS)
        with c4:
            native_country = st.selectbox("🏳️ Country", COUNTRY_OPTIONS)

        # ── Row 2 ──
        c1, c2 = st.columns(2)
        with c1:
            workclass = st.selectbox("🏢 Work Class", WORKCLASS_OPTIONS)
            education = st.selectbox("🎓 Education", EDUCATION_OPTIONS, index=12)
            marital_status = st.selectbox("💍 Marital Status", MARITAL_OPTIONS)
        with c2:
            occupation = st.selectbox("💼 Occupation", OCCUPATION_OPTIONS)
            relationship = st.selectbox("👥 Relationship", RELATIONSHIP_OPTIONS)
            hours_per_week = st.number_input("⏰ Hours/Week", min_value=1, max_value=99, value=40)

        # ── Row 3 ──
        c1, c2, c3 = st.columns(3)
        with c1:
            fnlwgt = st.number_input("📊 Final Weight (fnlwgt)", min_value=10000,
                                     max_value=1500000, value=200000)
        with c2:
            capital_gain = st.number_input("📈 Capital Gain ($)", min_value=0,
                                           max_value=100000, value=0)
        with c3:
            capital_loss = st.number_input("📉 Capital Loss ($)", min_value=0,
                                           max_value=4000, value=0)

        submitted = st.form_submit_button("🔮 Predict Income", use_container_width=True)

    if submitted:
        if not model_ready:
            st.error("❌ Model not trained. Please go to the **🚂 Train Model** tab first.")
        else:
            education_num = EDUCATION_NUM_MAP[education]
            input_data = {
                "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
                "education": education, "education_num": education_num,
                "marital_status": marital_status, "occupation": occupation,
                "relationship": relationship, "race": race, "sex": sex,
                "capital_gain": capital_gain, "capital_loss": capital_loss,
                "hours_per_week": hours_per_week, "native_country": native_country,
            }

            with st.spinner("Running prediction..."):
                time.sleep(0.5)
                pred, prob = make_prediction(model, preprocessor, input_data)

            st.divider()
            col_res, col_gauge = st.columns([2, 1])

            with col_res:
                if pred == 1:
                    st.markdown("""
                    <div class="result-high">
                        <div class="result-title">🟢 > $50,000 / year</div>
                        <div class="result-sub">This individual is predicted to earn <strong>more than $50K</strong> annually.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-low">
                        <div class="result-title">🔴 ≤ $50,000 / year</div>
                        <div class="result-sub">This individual is predicted to earn <strong>$50K or less</strong> annually.</div>
                    </div>
                    """, unsafe_allow_html=True)

                if prob is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence (>50K):** {prob[1]*100:.1f}%")
                    st.progress(float(prob[1]))

            with col_gauge:
                if prob is not None:
                    fig = render_gauge(prob[1])
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

            # ─ Feature summary ─
            with st.expander("📋 View Input Summary"):
                summary_df = pd.DataFrame([{
                    "Age": age, "Gender": sex, "Education": education,
                    "Occupation": occupation, "Hours/Week": hours_per_week,
                    "Marital Status": marital_status, "Capital Gain": f"${capital_gain:,}",
                    "Capital Loss": f"${capital_loss:,}", "Work Class": workclass,
                    "Country": native_country, "Race": race, "Relationship": relationship,
                }]).T.rename(columns={0: "Value"})
                st.dataframe(summary_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 · EDA
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 📊 Exploratory Data Analysis")

    df_raw = load_raw_data()
    if df_raw is None:
        st.info("📭 No data found. Train the model first to generate EDA plots.")
    else:
        # Quick stats
        df_clean = normalize_income_col(df_raw)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Total Records", f"{len(df_clean):,}")
        c2.metric("📐 Features", f"{df_clean.shape[1] - 1}")
        c3.metric("❓ Missing Values", f"{df_clean.isnull().sum().sum():,}")
        if "income" in df_clean.columns:
            pct_high = (df_clean["income"] == ">50K").mean() * 100
            c4.metric("💰 >50K Rate", f"{pct_high:.1f}%")

        st.divider()

        # Plot gallery
        plot_titles = {
            "01_target_distribution.png": "🎯 Target Distribution",
            "02_numerical_distributions.png": "📈 Numerical Feature Distributions",
            "03_correlation_heatmap.png": "🔥 Correlation Heatmap",
            "04_categorical_vs_target.png": "📊 Categorical Features vs Income",
            "05_age_analysis.png": "🎂 Age Analysis",
            "06_hours_analysis.png": "⏰ Hours per Week Analysis",
            "07_capital_analysis.png": "💵 Capital Gain / Loss Analysis",
            "08_education_analysis.png": "🎓 Education Level Analysis",
            "09_missing_values.png": "❓ Missing Values",
            "10_occupation_income.png": "💼 Occupation Income Rate",
        }

        eda_plots_exist = os.path.exists(EDA_DIR) and len(os.listdir(EDA_DIR)) > 0

        if not eda_plots_exist:
            with st.spinner("Generating EDA plots..."):
                sys.path.insert(0, os.getcwd())
                from src.components.eda import run_full_eda
                run_full_eda(df_raw, EDA_DIR)
            st.success("✅ EDA plots generated!")

        # Display plots in pairs
        plot_files = sorted([f for f in os.listdir(EDA_DIR) if f.endswith(".png")])
        for i in range(0, len(plot_files), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(plot_files):
                    fname = plot_files[i + j]
                    fpath = os.path.join(EDA_DIR, fname)
                    title = plot_titles.get(fname, fname.replace("_", " ").replace(".png", "").title())
                    with col:
                        st.markdown(f"**{title}**")
                        try:
                            img = Image.open(fpath)
                            st.image(img, use_container_width=True)
                        except Exception:
                            st.warning(f"Could not load {fname}")

        # Raw stats
        with st.expander("🔬 Descriptive Statistics"):
            tab_a, tab_b = st.tabs(["Numerical", "Categorical"])
            with tab_a:
                num_df = df_clean.select_dtypes(include=np.number)
                st.dataframe(num_df.describe().round(2), use_container_width=True)
            with tab_b:
                cat_df = df_clean.select_dtypes(include="object")
                st.dataframe(cat_df.describe(), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 · MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 🏆 Model Performance Dashboard")

    report = load_model_report()
    if report is None:
        st.info("📭 No model report found. Train the model first.")
    else:
        best_name = report.get("best_model", "")
        models_data = report.get("models", {})

        # ─ Best model banner ─
        if best_name in models_data:
            bi = models_data[best_name]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15));
                        border: 1px solid rgba(99,102,241,0.5); border-radius:16px; padding:1.5rem; margin-bottom:1.5rem;">
                <h3 style="margin:0; color:#a78bfa;">🥇 Best Model: {best_name}</h3>
                <p style="margin:0.5rem 0 0; color:#ddd6fe; font-family:monospace; font-size:0.95rem;">
                    Best params: {json.dumps(bi.get('best_params', {}), indent=0)}
                </p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🎯 Accuracy", f"{bi['accuracy']*100:.2f}%")
            c2.metric("📐 Precision", f"{bi['precision']*100:.2f}%")
            c3.metric("🔁 Recall", f"{bi['recall']*100:.2f}%")
            c4.metric("⚖️ F1 Score", f"{bi['f1_score']*100:.2f}%")
            c5.metric("📈 ROC-AUC", f"{bi.get('roc_auc', 0)*100:.2f}%")

        st.divider()

        # ─ All models comparison ─
        st.markdown("#### 📊 All Models Comparison")
        rows = []
        for name, info in models_data.items():
            rows.append({
                "Model": name,
                "Accuracy": info["accuracy"],
                "Precision": info["precision"],
                "Recall": info["recall"],
                "F1 Score": info["f1_score"],
                "ROC-AUC": info.get("roc_auc", None),
                "CV F1 Mean": info["cv_f1_mean"],
                "CV F1 Std": info["cv_f1_std"],
                "Best": "⭐" if name == best_name else "",
            })

        comp_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)

        def highlight_best(row):
            if row["Best"] == "⭐":
                return ["background-color: rgba(99,102,241,0.2)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            comp_df.style.apply(highlight_best, axis=1)
                   .format({c: "{:.4f}" for c in ["Accuracy","Precision","Recall",
                                                    "F1 Score","ROC-AUC","CV F1 Mean","CV F1 Std"]
                             if c in comp_df.columns}),
            use_container_width=True, height=380
        )

        # ─ Bar chart comparison ─
        st.markdown("#### 📈 Visual Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#0f0f1a")

        metrics_to_plot = ["F1 Score", "ROC-AUC"]
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(comp_df)))

        for ax, metric in zip(axes, metrics_to_plot):
            if metric in comp_df.columns and comp_df[metric].notna().any():
                bars = ax.barh(comp_df["Model"], comp_df[metric],
                               color=colors, edgecolor="none", height=0.6)
                ax.bar_label(bars, fmt="%.3f", padding=3, color="white", fontsize=9)
                ax.set_title(metric, color="white", fontweight="bold")
                ax.set_facecolor("#0f0f1a")
                ax.tick_params(colors="white")
                ax.spines[:].set_visible(False)
                ax.set_xlim(0, 1.1)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color("white")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ─ Confusion Matrix ─
        if best_name in models_data and "confusion_matrix" in models_data[best_name]:
            st.markdown("#### 🔲 Confusion Matrix (Best Model)")
            cm = np.array(models_data[best_name]["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#0f0f1a")
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="magma",
                xticklabels=["≤50K", ">50K"],
                yticklabels=["≤50K", ">50K"],
                ax=ax, linewidths=1,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title(f"Confusion Matrix — {best_name}", color="white", fontweight="bold")
            ax.set_xlabel("Predicted", color="white")
            ax.set_ylabel("Actual", color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig, use_container_width=False)
            plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 · TRAIN MODEL
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🚂 Train the ML Pipeline")
    st.markdown("""
    This will execute the full pipeline:
    1. **Data Ingestion** — Download Adult Census Income dataset
    2. **Data Transformation** — Clean, engineer features, encode & scale
    3. **Model Training** — Train 9 classifiers with GridSearchCV
    4. **Model Selection** — Pick the best model by F1 score
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader(
            "📁 Upload custom dataset (optional)",
            type=["csv"],
            help="Leave blank to use the built-in Adult Census Income dataset."
        )
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        start_training = st.button("🚀 Start Training Pipeline", use_container_width=True)

    trigger = st.session_state.pop("trigger_training", False)

    if start_training or trigger:
        import sys as _sys
        _sys.path.insert(0, os.getcwd())

        # Save uploaded file if provided
        custom_path = None
        if uploaded_file:
            custom_path = os.path.join(ARTIFACTS_DIR, "uploaded_data.csv")
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            with open(custom_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"📁 Using uploaded file: {uploaded_file.name}")

        progress = st.progress(0, text="Initializing...")
        status = st.empty()

        try:
            status.info("📥 Step 1/3: Data Ingestion...")
            progress.progress(10, "Downloading data...")
            from src.components.data_ingestion import DataIngestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion(custom_path)
            progress.progress(30, "Data ingested ✅")

            status.info("⚙️ Step 2/3: Data Transformation...")
            from src.components.data_transformation import DataTransformation
            transformation = DataTransformation()
            X_train, y_train, X_test, y_test, _ = \
                transformation.initiate_data_transformation(train_path, test_path)
            progress.progress(50, "Data transformed ✅")

            status.info("🤖 Step 3/3: Training Models (this may take a few minutes)...")
            from src.components.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            best_name, best_info, full_report = trainer.initiate_model_training(
                X_train, y_train, X_test, y_test
            )
            progress.progress(90, "Models trained ✅")

            # Generate EDA plots
            df_raw_new = pd.read_csv(os.path.join(ARTIFACTS_DIR, "raw.csv"))
            from src.components.eda import run_full_eda
            run_full_eda(df_raw_new, EDA_DIR)
            progress.progress(100, "Complete ✅")

            status.success(f"🎉 Training complete! Best model: **{best_name}**")
            st.balloons()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏆 Best Model", best_name)
            c2.metric("⚖️ F1 Score", f"{best_info['f1_score']*100:.2f}%")
            c3.metric("🎯 Accuracy", f"{best_info['accuracy']*100:.2f}%")
            c4.metric("📈 ROC-AUC", f"{best_info.get('roc_auc', 0)*100:.2f}%")

            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        except Exception as e:
            progress.empty()
            status.error(f"❌ Training failed: {str(e)}")
            st.exception(e)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 · DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 📋 Data Explorer")

    df_raw = load_raw_data()
    if df_raw is None:
        st.info("📭 No data loaded yet. Train the model to load data.")
    else:
        df_show = normalize_income_col(df_raw)

        # Filters
        with st.expander("🔍 Filters", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                if "income" in df_show.columns:
                    income_filter = st.multiselect("Income", df_show["income"].unique(),
                                                   default=list(df_show["income"].unique()))
                else:
                    income_filter = None
            with fc2:
                if "sex" in df_show.columns:
                    sex_vals = df_show["sex"].astype(str).str.strip().unique().tolist()
                    sex_filter = st.multiselect("Sex", sex_vals, default=sex_vals)
                else:
                    sex_filter = None
            with fc3:
                if "age" in df_show.columns:
                    age_range = st.slider("Age Range", int(df_show["age"].min()),
                                          int(df_show["age"].max()),
                                          (int(df_show["age"].min()), int(df_show["age"].max())))

        filtered = df_show.copy()
        if income_filter and "income" in filtered.columns:
            filtered = filtered[filtered["income"].isin(income_filter)]
        if sex_filter and "sex" in filtered.columns:
            filtered = filtered[filtered["sex"].astype(str).str.strip().isin(sex_filter)]
        if "age" in filtered.columns:
            filtered = filtered[
                (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])
            ]

        st.markdown(f"**Showing {len(filtered):,} of {len(df_show):,} records**")
        st.dataframe(filtered.head(500), use_container_width=True, height=400)

        # Download
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Data",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv",
        )

#.  streamlit run app_streamlit.py
#.  source .venv/bin/activate 