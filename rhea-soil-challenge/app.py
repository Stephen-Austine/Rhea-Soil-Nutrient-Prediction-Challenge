import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============= NATURE/EARTH THEME CSS =============
NATURE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

/* Global Nature Theme Base */
.stApp {
    background: linear-gradient(180deg, #f5f7f0 0%, #e8efe0 50%, #dce7d0 100%);
    min-height: 100vh;
    color: #2d3a2d;
    font-family: 'Nunito', 'Segoe UI', sans-serif;
    max-width: 1400px;
    margin: 0 auto;
}

/* Main container padding */
.main-content {
    padding: 0 20px;
    max-width: 1200px;
    margin: 0 auto;
}

/* Nature-Inspired Glass Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(139, 195, 74, 0.25);
    border-radius: 24px;
    padding: 28px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(76, 122, 37, 0.08);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    box-shadow: 0 12px 48px rgba(76, 122, 37, 0.12);
    border-color: rgba(139, 195, 74, 0.45);
    transform: translateY(-2px);
}

/* Nature Header Gradient Bar */
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 24px;
    right: 24px;
    height: 4px;
    background: linear-gradient(90deg, #4caf50, #8bc34a, #cddc39);
    border-radius: 24px 24px 0 0;
    opacity: 0.95;
}

/* Position relative for cards */
.glass-card {
    position: relative;
}

/* Nature-Inspired Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4caf50 0%, #66bb6a 50%, #8bc34a 100%);
    color: #ffffff;
    border: none;
    border-radius: 14px;
    padding: 14px 28px;
    font-weight: 700;
    font-size: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton>button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
}

.stButton>button:hover::before {
    left: 100%;
}

.stButton>button:active {
    transform: translateY(0);
}

/* Primary Button Variant */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #388e3c 0%, #4caf50 50%, #66bb6a 100%);
}

/* Nature Input Fields */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>select,
.stTextArea>div>div>textarea {
    background: rgba(255, 255, 255, 0.95);
    border: 2px solid rgba(139, 195, 74, 0.3);
    border-radius: 14px;
    color: #2d3a2d;
    padding: 14px 18px;
    font-size: 1rem;
    transition: all 0.3s ease;
}

.stTextInput>div>div>input:focus,
.stNumberInput>div>div>input:focus,
.stSelectbox>div>div>select:focus,
.stTextArea>div>div>textarea:focus {
    background: #ffffff;
    border-color: rgba(76, 175, 80, 0.6);
    box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.1);
    outline: none;
}

/* Headers - Nature Style */
h1, h2, h3, h4, h5, h6 {
    color: #2e5a2e;
    font-weight: 800;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.8rem;
    background: linear-gradient(135deg, #2e7d32, #388e3c, #4caf50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

h2 {
    font-size: 1.8rem;
    color: #33691e;
    font-weight: 700;
}

h3 {
    font-size: 1.4rem;
    color: #558b2f;
}

/* Nature Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.12), rgba(139, 195, 74, 0.08));
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    border: 1px solid rgba(76, 175, 80, 0.25);
    transition: all 0.3s ease;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(76, 175, 80, 0.15);
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.18), rgba(139, 195, 74, 0.12));
}

.metric-value {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2e7d32, #43a047);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 12px 0;
}

.metric-label {
    font-size: 1rem;
    color: #558b2f;
    font-weight: 700;
    letter-spacing: 0.02em;
}

/* Progress Bars - Nature Style */
.stProgress>div>div>div>div {
    background: linear-gradient(90deg, #4caf50, #8bc34a, #cddc39);
    border-radius: 12px;
    height: 10px;
    box-shadow: 0 0 15px rgba(76, 175, 80, 0.3);
}

/* Sidebar - Forest Theme */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f5e9 0%, #c8e6c9 100%);
    border-right: 1px solid rgba(76, 175, 80, 0.2);
    padding-top: 20px;
}

/* Section Styling */
.section {
    margin: 20px 0;
    padding: 20px;
}

/* Tab Styling - Nature */
[data-testid="stTabs"] {
    margin: 24px 0;
}

[data-testid="stTabs"] [role="tab"] {
    background: rgba(255, 255, 255, 0.7);
    color: #558b2f;
    border: 2px solid transparent;
    border-radius: 16px 16px 0 0;
    padding: 16px 28px;
    margin-right: 8px;
    transition: all 0.3s ease;
    font-weight: 700;
    font-size: 1rem;
}

[data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(76, 175, 80, 0.15);
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(180deg, rgba(76, 175, 80, 0.2), rgba(139, 195, 74, 0.1));
    color: #2e7d32;
    border-color: rgba(76, 175, 80, 0.3);
    transform: translateY(-3px);
}

/* DataFrame/Table Styling */
.stDataFrame {
    background: rgba(255, 255, 255, 0.9);
    border: 2px solid rgba(139, 195, 74, 0.2);
    border-radius: 16px;
    overflow: hidden;
}

/* Success Message */
.stSuccess {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(139, 195, 74, 0.1));
    border: 1px solid rgba(76, 175, 80, 0.4);
    border-radius: 14px;
    padding: 18px;
    color: #2e7d32;
}

/* Error Message */
.stError {
    background: linear-gradient(135deg, rgba(211, 47, 47, 0.1), rgba(244, 67, 54, 0.08));
    border: 1px solid rgba(211, 47, 47, 0.3);
    border-radius: 14px;
    padding: 18px;
    color: #c62828;
}

/* Warning Message */
.stWarning {
    background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(255, 193, 7, 0.08));
    border: 1px solid rgba(255, 152, 0, 0.3);
    border-radius: 14px;
    padding: 18px;
    color: #e65100;
}

/* Info Message */
.stInfo {
    background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(25, 118, 210, 0.08));
    border: 1px solid rgba(33, 150, 243, 0.3);
    border-radius: 14px;
    padding: 18px;
    color: #1565c0;
}

/* Spinner */
.stSpinner > div {
    border-color: rgba(76, 175, 80, 0.3);
    border-top-color: #4caf50;
}

/* Responsive Design */
@media (max-width: 768px) {
    .stApp {
        padding: 12px;
        max-width: 100%;
    }
    
    .glass-card {
        padding: 20px;
        margin: 12px 0;
    }
    
    .metric-card {
        padding: 18px;
        margin: 8px 0;
    }
    
    .metric-value {
        font-size: 2rem;
    }
    
    h1 {
        font-size: 2.2rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
    
    [data-testid="stTabs"] [role="tab"] {
        padding: 12px 18px;
        font-size: 0.9rem;
        margin-right: 4px;
    }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(76, 175, 80, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(76, 175, 80, 0.4);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(76, 175, 80, 0.6);
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.3), transparent);
    margin: 20px 0;
}

/* Two column layout helper */
.cols-equal-gap > div {
    gap: 1.5rem;
}

/* Card grid */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
}

/* Chart container */
.chart-container {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 16px;
    padding: 16px;
    margin: 12px 0;
}

/* Status message */
.status-message {
    padding: 16px 20px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 12px 0;
}

/* Help box */
.help-box {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(139, 195, 74, 0.08));
    padding: 24px;
    border-radius: 20px;
    margin: 16px 0;
    border: 2px solid rgba(76, 175, 80, 0.2);
}
</style>
"""

st.markdown(NATURE_CSS, unsafe_allow_html=True)

# ============= HEADER SECTION =============
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Header with title and help button
col_title, col_help = st.columns([9, 1])
with col_title:
    st.markdown("""
    <div style="padding: 20px 0 12px 0;">
        <h1 style="font-size: 2.9rem; margin-bottom: 8px;">🌱 Rhea Soil Nutrient Predictor</h1>
        <p style="font-size: 1.2rem; color: #558b2f; margin: 0; font-weight: 600;">
            Predict 13 soil nutrients where lab tests aren't available
        </p>
    </div>
    """, unsafe_allow_html=True)
with col_help:
    st.markdown("<div style='padding-top: 24px;'>", unsafe_allow_html=True)
    if st.button("❓", key="help_toggle", help="Toggle help information"):
        st.session_state.show_help = not st.session_state.get('show_help', False)
    st.markdown("</div>", unsafe_allow_html=True)

# Help section (collapsible)
if st.session_state.get('show_help', False):
    st.markdown("""
    <div class="help-box">
        <h3 style="color: #2e7d32; margin-top: 0; margin-bottom: 14px;">📖 How to use</h3>
        <ul style="color: #33691e; line-height: 1.8; padding-left: 24px; margin: 0 0 14px 0; font-size: 1.05rem;">
            <li>Upload your datasets or use the preloaded sample data</li>
            <li>Configure model parameters in the Train tab</li>
            <li>Train predictive models for all 13 soil nutrients</li>
            <li>Generate a competition-ready submission file with zero-mask applied</li>
        </ul>
        <div style="background: rgba(255, 152, 0, 0.12); padding: 14px; border-radius: 14px; border-left: 4px solid #ff9800;">
            <strong style="color: #e65100;">⚠️ Important:</strong> Predictions for entries marked 0 in TargetPred_To_Keep.csv will be automatically set to 0 to comply with competition rules.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============= LOAD DATA =============
@st.cache_data
def load_sample_data():
    """Load data from current directory"""
    try:
        train = pd.read_csv("Train.csv")
        test = pd.read_csv("TestSet.csv")
        mask = pd.read_csv("TargetPred_To_Keep.csv")
        sub_template = pd.read_csv("SampleSubmission.csv")
        return train, test, mask, sub_template
    except Exception as e:
        return None, None, None, None

train, test, mask, sub_template = load_sample_data()

# Data loading status
if train is not None:
    st.markdown(f"""
    <div class="status-message" style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.12), rgba(139, 195, 74, 0.08)); border: 1px solid rgba(76, 175, 80, 0.25);">
        <span style="font-size: 1.6rem;">✅</span>
        <div>
            <div style="font-weight: 700; color: #2e7d32; font-size: 1.1rem;">All datasets loaded successfully</div>
            <div style="color: #558b2f; font-size: 0.95rem;">Train: <strong>{len(train):,}</strong> samples | Test: <strong>{len(test):,}</strong> samples</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-message" style="background: linear-gradient(135deg, rgba(211, 47, 47, 0.08), rgba(244, 67, 54, 0.06)); border: 1px solid rgba(211, 47, 47, 0.25);">
        <span style="font-size: 1.6rem; color: #c62828;">⚠️</span>
        <div>
            <div style="font-weight: 700; color: #c62828; font-size: 1.1rem;">Could not load datasets</div>
            <div style="color: #e57373; font-size: 0.95rem;">Place CSV files in the same folder as app.py</div>
            <div style="color: #ef9a9a; font-size: 0.85rem; margin-top: 4px;">Required: Train.csv, TestSet.csv, TargetPred_To_Keep.csv, SampleSubmission.csv</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============= DASHBOARD METRICS =============
if train is not None:
    st.markdown("### 📊 Dataset Overview", unsafe_allow_html=True)
    
    # Metrics in a balanced row
    m1, m2, m3, m4 = st.columns(4)
    metrics_data = [
        ("📊 Train Samples", f"{len(train):,}"),
        ("🎯 Test Samples", f"{len(test):,}"),
        ("🧪 Nutrients", "13"),
        ("⏰ Deadline", "7 days")
    ]
    
    for col, (label, val) in zip([m1, m2, m3, m4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

# ============= MAIN TABS =============
tab1, tab2, tab3, tab4 = st.tabs(["📊 Explore Data", "🔬 Train Models", "📤 Generate Submission", "⚙️ Settings"])

# ============= TAB 1: EXPLORE =============
with tab1:
    if train is not None:
        # Row 1: Two charts side by side
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🗺️ Geographic Distribution")
            
            sample_size = min(2000, len(train))
            train_sample = train.sample(sample_size, random_state=42)
            nutrient_cols = ['N', 'P', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn', 'B', 'Na', 'Al', 'S']
            color_col = 'N' if 'N' in train_sample.columns else train_sample.columns[2]
            
            fig = px.scatter_mapbox(
                train_sample,
                lat="Latitude", lon="Longitude", color=color_col,
                zoom=3, height=380, mapbox_style="open-street-map",
                color_continuous_scale="Greens"
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(255,255,255,0.5)',
                font=dict(color='#2d3a2d')
            )
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📈 Nutrient Correlations")
            
            available_nutrients = [c for c in nutrient_cols if c in train.columns]
            if available_nutrients:
                corr = train[available_nutrients].corr().round(2)
                fig = px.imshow(corr, color_continuous_scale="Greens", aspect="auto")
                fig.update_layout(
                    height=380,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(255,255,255,0.5)',
                    font=dict(color='#2d3a2d')
                )
                st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Data completeness
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔍 Data Completeness")
        
        available_nutrients = [c for c in nutrient_cols if c in train.columns]
        if available_nutrients:
            missing = train[available_nutrients].isnull().sum().sort_values(ascending=False)
            fig = px.bar(
                x=missing.values, y=missing.index, orientation='h',
                labels={'x': 'Missing Values', 'y': 'Nutrient'},
                color=missing.values, color_continuous_scale="Greens"
            )
            fig.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(255,255,255,0.5)',
                font=dict(color='#2d3a2d'),
                plot_bgcolor='rgba(255,255,255,0.9)'
            )
            st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please ensure data files are available to explore the dataset.")

# ============= TAB 2: TRAIN =============
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Configuration")
    
    # Configuration in 3 columns
    cfg1, cfg2, cfg3 = st.columns(3)
    
    with cfg1:
        model_type = st.selectbox(
            "Algorithm",
            ["XGBoost", "LightGBM", "Random Forest"],
            index=0
        )
        cv_folds = st.slider("CV Folds", 3, 10, 5)
    
    with cfg2:
        max_depth = st.slider("Max Depth", 3, 15, 8)
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
    
    with cfg3:
        n_est = st.slider("Estimators", 50, 500, 200, step=50)
        use_geo = st.checkbox("Geospatial Features", value=True)
    
    st.markdown("---")
    
    # Nutrient selection
    st.markdown("**Select nutrients to predict:**")
    all_nutrients = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']
    selected = st.multiselect(
        "Nutrients",
        all_nutrients,
        default=all_nutrients,
        label_visibility="collapsed"
    )
    
    # Train button
    st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
    train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if train_btn:
        if train is None:
            st.error("Please load data first")
        else:
            with st.spinner("Training models... 🌱"):
                progress = st.progress(0)
                
                # Try to import from main
                try:
                    from main import SoilNutrientPredictor
                    
                    model_type_lower = model_type.lower().replace(' ', '_')
                    
                    predictor = SoilNutrientPredictor(
                        model_type=model_type_lower,
                        cv_folds=cv_folds,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_est
                    )
                    
                    predictor.load_data(train, test, mask)
                    results = predictor.train(selected)
                    
                    st.success("✅ Training complete!")
                    
                    # Display results
                    st.markdown("### 📊 Training Results")
                    
                    # Results in a nice grid
                    for nutrient, metrics in results.items():
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(139, 195, 74, 0.06));
                                    padding: 16px; border-radius: 12px; margin: 8px 0;
                                    border: 1px solid rgba(76, 175, 80, 0.2);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 700; color: #2e7d32; font-size: 1.1rem;">🧪 {nutrient}</div>
                                <div style="color: #33691e;">CV RMSE: <strong style="color: #43a047;">{metrics['cv_rmse']:.3f}</strong> ± {metrics['cv_std']:.3f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Save models
                    predictor.save("models/")
                    st.session_state.predictor = predictor
                    st.session_state.trained = True
                    progress.progress(100)
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.info("Make sure main.py contains the SoilNutrientPredictor class.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= TAB 3: SUBMIT =============
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📤 Generate Submission")
    
    st.warning("⚠️ Predictions for TargetPred_To_Keep entries marked 0 will be auto-set to 0")
    
    if st.session_state.get('trained', False):
        # Generate button
        gen_btn = st.button("🔮 Generate Predictions", type="primary", use_container_width=True)
        
        if gen_btn:
            with st.spinner("Generating predictions... 🌾"):
                try:
                    submission = st.session_state.predictor.predict()
                    st.session_state.submission = submission
                    
                    # Preview
                    st.markdown("### 📊 Submission Preview")
                    st.dataframe(submission.head(10), width='stretch')
                    
                    # Validation
                    from utils import validate_submission
                    valid, errors = validate_submission(submission, all_nutrients)
                    
                    if valid:
                        st.markdown("""
                        <div class="status-message" style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.12), rgba(139, 195, 74, 0.08)); border: 1px solid rgba(76, 175, 80, 0.25);">
                            <span style="font-size: 1.4rem;">✅</span>
                            <strong style="color: #2e7d32;">Submission format valid!</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-message" style="background: linear-gradient(135deg, rgba(211, 47, 47, 0.08), rgba(244, 67, 54, 0.06)); border: 1px solid rgba(211, 47, 47, 0.25);">
                            <span style="font-size: 1.4rem;">❌</span>
                            <strong style="color: #c62828;">Validation Issues: {', '.join(errors)}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download
                    csv = submission.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download submission.csv",
                        data=csv,
                        file_name="submission.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 48px 24px; background: linear-gradient(135deg, rgba(76, 175, 80, 0.08), rgba(139, 195, 74, 0.06));
                    border-radius: 16px; border: 1px solid rgba(76, 175, 80, 0.15); margin: 24px 0;">
            <div style="font-size: 3rem; margin-bottom: 16px;">👈</div>
            <div style="font-size: 1.2rem; color: #558b2f; margin-bottom: 8px; font-weight: 600;">Train models first in the 'Train Models' tab</div>
            <div style="font-size: 1rem; color: #7cb342;">Once trained, you can generate your submission file here</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Format reference
    st.markdown("---")
    st.markdown("### 📋 Submission Format Reference")
    st.code("ID,Target_Al,Target_B,Target_Ca,...,Target_Zn\nID_XYZ,12.34,5.67,...,2.34", language="csv")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= TAB 4: SETTINGS =============
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Preferences")
    
    # Visual Design Section
    st.markdown("### 🎨 Visual Design")
    v1, v2 = st.columns(2)
    with v1:
        st.checkbox("Nature theme (green/earth tones)", value=True, disabled=True)
    with v2:
        st.checkbox("Glassmorphism cards", value=True)
    
    # Layout Section
    st.markdown("### 📐 Layout")
    l1, l2 = st.columns(2)
    with l1:
        st.checkbox("Top-down data flow", value=True)
    with l2:
        st.checkbox("Reset inputs after use", value=True)
    
    # Actions Section
    st.markdown("### 🛠️ Actions")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("💾 Save Preferences", use_container_width=True):
            st.toast("Preferences saved 🌿", icon="✅")
    with a2:
        if st.button("🔄 Reset All Fields", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['trained', 'predictor', 'submission']:
                    del st.session_state[key]
            st.toast("Fields reset 🔄", icon="🔄")
    
    # About Section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(139, 195, 74, 0.06));
                padding: 20px; border-radius: 16px; border: 1px solid rgba(76, 175, 80, 0.2);">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <span style="font-size: 2rem;">🌱</span>
            <strong style="color: #2e7d32; font-size: 1.2rem;">Rhea Soil Nutrient Predictor</strong>
        </div>
        <div style="color: #558b2f;">Version 2.0 | Built for the Rhea Soil Challenge</div>
        <div style="color: #7cb342; margin-top: 4px;">Powered by XGBoost, LightGBM, and Random Forest</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============= FOOTER =============
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 24px; color: #558b2f;">
    <div style="font-weight: 700; margin-bottom: 6px;">🌱 Rhea Soil Nutrient Prediction | Built for African Agriculture</div>
    <div style="color: #7cb342; font-size: 0.95rem;"><em>Open-source • Reproducible • Competition-ready</em></div>
    <div style="color: #8bc34a; font-size: 0.85rem; margin-top: 6px;">© 2026 | Powered by Machine Learning 🌾</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

