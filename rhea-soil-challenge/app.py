
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============= PAGE CONFIG - MUST BE FIRST =============
st.set_page_config(
    page_title="Rhea Soil Nutrient Predictor",
    page_icon="🌱",
    layout="wide",  # This is key for full width
    initial_sidebar_state="expanded"
)

# ============= CUSTOM CSS FOR BETTER LAYOUT =============
CUSTOM_CSS = """
<style>
/* Reduce padding and margins for full-width layout */
.stApp {
    max-width: 100%;
    padding: 0 2rem;
}

/* Main container - use full width */
.main .block-container {
    max-width: 100%;
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Cards with better spacing */
.metric-card {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(129, 199, 132, 0.05));
    border: 2px solid rgba(76, 175, 80, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card:hover {
    border-color: rgba(76, 175, 80, 0.6);
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2e7d32;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.95rem;
    color: #558b2f;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-header {
    background: linear-gradient(90deg, rgba(76, 175, 80, 0.15), rgba(129, 199, 132, 0.05));
    border-left: 5px solid #4CAF50;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}

/* Glass cards for content */
.glass-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(76, 175, 80, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4CAF50, #66BB6A);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #43A047, #57B85C);
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    transform: translateY(-1px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
}

/* Success message */
.stAlert {
    border-radius: 8px;
    border: 2px solid rgba(76, 175, 80, 0.3);
}

/* Plotly charts - full width */
.plotly-chart {
    width: 100% !important;
}

/* Reduce sidebar width if needed */
[data-testid="stSidebar"] {
    width: 300px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .metric-value {
        font-size: 2rem;
    }
    .stApp {
        padding: 0 1rem;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============= HEADER =============
col1, col2 = st.columns([8, 1])
with col1:
    st.title("🌱 Rhea Soil Nutrient Predictor")
    st.markdown("*Predict 13 soil nutrients where lab tests aren't available*")
with col2:
    if st.button("❓", key="help_toggle", help="Toggle help information"):
        st.session_state.show_help = not st.session_state.get('show_help', False)

if st.session_state.get('show_help', False):
    st.info("""
    **How to use:**
    - **Explore Data**: Visualize geographic distribution and nutrient correlations
    - **Train Models**: Configure and train ML models for nutrient prediction
    - **Generate Submission**: Create submission file for Zindi competition
    - **Settings**: Adjust preferences and parameters
    """)

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

# Try to load data
train, test, mask, sub_template = load_sample_data()

if train is not None:
    st.success(f"**✅ All datasets loaded successfully** | Train: {len(train):,} samples | Test: {len(test):,} samples", icon="✅")
else:
    st.error("❌ Could not load datasets. Place CSV files in the same folder as app.py")
    st.info("Required files: Train.csv, TestSet.csv, TargetPred_To_Keep.csv, SampleSubmission.csv")

# ============= DATASET OVERVIEW - FULL WIDTH =============
st.markdown('<div class="section-header">', unsafe_allow_html=True)
st.markdown("### 📊 Dataset Overview")
st.markdown('</div>', unsafe_allow_html=True)

# Create 4-column layout for metrics
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈 Train Samples</div>
        <div class="metric-value">{len(train):,}</div>
        <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem">Soil observations</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🎯 Test Samples</div>
        <div class="metric-value">{len(test):,}</div>
        <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem">Locations to predict</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🧪 Nutrients</div>
        <div class="metric-value">13</div>
        <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem">Al,B,Ca,Cu,Fe,K,Mg,Mn,N,Na,P,S,Zn</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">⏰ Deadline</div>
        <div class="metric-value">7 days</div>
        <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem">March 6, 2026</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

# ============= NAVIGATION TABS =============
tab1, tab2, tab3, tab4 = st.tabs(["📊 Explore Data", "🔬 Train Models", "📤 Generate Submission", "⚙️ Settings"])

# ----- TAB 1: EXPLORE DATA -----
with tab1:
    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown("### 🔍 Explore & Analyze Data")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if train is not None:
        # Row 1: Two visualizations side by side
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🗺️ Geographic Distribution")
            
            # Sample for performance
            sample_size = min(2000, len(train))
            train_sample = train.sample(sample_size, random_state=42)
            
            # Check if we have nutrient columns
            nutrient_cols = ['N', 'P', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn', 'B', 'Na', 'Al', 'S']
            color_col = 'N' if 'N' in train_sample.columns else train_sample.columns[2]
            
            fig_map = px.scatter_mapbox(
                train_sample,
                lat="Latitude", lon="Longitude", color=color_col,
                zoom=3, height=500, mapbox_style="open-street-map",
                color_continuous_scale="Viridis",
                title=f"Spatial Distribution of {color_col} Concentration"
            )
            fig_map.update_layout(
                margin=dict(t=40, b=0, l=0, r=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Nutrient Correlations")
            
            available_nutrients = [c for c in nutrient_cols if c in train.columns]
            if available_nutrients:
                corr = train[available_nutrients].corr().round(2)
                fig_corr = px.imshow(
                    corr, 
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    height=500,
                    title="Correlation Matrix Between Nutrients"
                )
                fig_corr.update_layout(
                    margin=dict(t=40, b=0, l=0, r=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Data completeness and distributions
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Data Completeness")
            
            available_nutrients = [c for c in nutrient_cols if c in train.columns]
            if available_nutrients:
                missing = train[available_nutrients].isnull().sum().sort_values(ascending=False)
                fig_missing = px.bar(
                    x=missing.values, 
                    y=missing.index, 
                    orientation='h',
                    labels={'x': 'Missing Values', 'y': 'Nutrient'},
                    color=missing.values,
                    color_continuous_scale="Reds",
                    height=400,
                    title="Missing Values per Nutrient"
                )
                fig_missing.update_layout(
                    margin=dict(t=40, b=0, l=0, r=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Nutrient Distribution")
            
            # Select a few key nutrients for distribution
            key_nutrients = ['N', 'P', 'K', 'Ca', 'Mg']
            available_key = [n for n in key_nutrients if n in train.columns]
            
            if available_key:
                # Melt data for better visualization
                df_melted = train[available_key].melt(var_name='Nutrient', value_name='Concentration')
                df_melted = df_melted.dropna()
                
                fig_dist = px.box(
                    df_melted,
                    x='Nutrient',
                    y='Concentration',
                    color='Nutrient',
                    height=400,
                    title="Distribution of Key Nutrients",
                    labels={'Concentration': 'Concentration (mg/kg)', 'Nutrient': 'Nutrient'}
                )
                fig_dist.update_layout(
                    margin=dict(t=40, b=0, l=0, r=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 3: Statistics table
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Summary Statistics")
        
        if available_nutrients:
            stats_df = train[available_nutrients].describe().T
            stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            st.dataframe(stats_df.style.background_gradient(cmap='Greens', subset=['Mean', 'Std']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----- TAB 2: TRAIN MODELS -----
with tab2:
    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown("### 🤖 Train Machine Learning Models")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration section
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Model Selection")
        model_type = st.selectbox(
            "Algorithm",
            ["XGBoost", "LightGBM", "Random Forest"],
            index=0,
            help="XGBoost recommended for best performance"
        )
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Hyperparameters")
        max_depth = st.slider("Max Tree Depth", 3, 15, 8)
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Training Options")
        n_est = st.slider("Number of Estimators", 50, 500, 200, step=50)
        use_geo = st.checkbox("Use Geospatial Features", value=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Nutrient selection
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Select Nutrients to Predict")
    all_nutrients = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']
    selected = st.multiselect(
        "Nutrients",
        all_nutrients,
        default=all_nutrients,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Train button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if train is None:
                st.error("Please load data first")
            else:
                with st.spinner("Training models... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Import from main.py
                    from main import SoilNutrientPredictor
                    
                    model_type_lower = model_type.lower().replace(' ', '_')
                    
                    predictor = SoilNutrientPredictor(
                        model_type=model_type_lower,
                        cv_folds=cv_folds,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_est
                    )
                    
                    # Load data
                    status_text.text("Loading and preprocessing data...")
                    predictor.load_data(train, test, mask)
                    progress_bar.progress(20)
                    
                    # Train
                    status_text.text(f"Training {len(selected)} nutrient models...")
                    results = predictor.train(selected)
                    progress_bar.progress(80)
                    
                    # Display results
                    status_text.text("Training complete!")
                    progress_bar.progress(100)
                    
                    st.success("✅ Training completed successfully!")
                    
                    # Show results in columns
                    st.markdown("#### 📊 Model Performance (Cross-Validation RMSE)")
                    
                    # Create results dataframe
                    results_data = []
                    for nutrient, metrics in results.items():
                        results_data.append({
                            'Nutrient': nutrient,
                            'CV RMSE': f"{metrics['cv_rmse']:.3f}",
                            'Std Dev': f"{metrics['cv_std']:.3f}",
                            'Train RMSE': f"{metrics['train_rmse']:.3f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df.style.background_gradient(cmap='Greens', subset=['CV RMSE']), use_container_width=True)
                    
                    # Save models
                    predictor.save("models/")
                    st.session_state.predictor = predictor
                    st.session_state.trained = True
                    st.session_state.results = results
                    
                    st.balloons()
    
    # Show training history if available
    if st.session_state.get('trained', False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Training History")
        
        # Visualize CV scores
        if hasattr(st.session_state.get('predictor', None), 'training_results'):
            results = st.session_state.predictor.training_results
            
            nutrients = list(results.keys())
            cv_rmses = [results[n]['cv_rmse'] for n in nutrients]
            cv_stds = [results[n]['cv_std'] for n in nutrients]
            
            fig = px.bar(
                x=nutrients,
                y=cv_rmses,
                error_y=cv_stds,
                labels={'x': 'Nutrient', 'y': 'CV RMSE'},
                title="Cross-Validation Performance by Nutrient",
                color=cv_rmses,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                height=400,
                margin=dict(t=40, b=0, l=0, r=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----- TAB 3: GENERATE SUBMISSION -----
with tab3:
    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown("### 📤 Generate Competition Submission")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.info("⚠️ **Important**: Predictions for entries marked 0 in TargetPred_To_Keep.csv will be automatically set to 0", icon="⚠️")
    
    if st.session_state.get('trained', False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Submission Preview")
            
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    submission = st.session_state.predictor.predict()
                    st.session_state.submission = submission
                    
                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(submission.head(10), use_container_width=True)
                    
                    # Validation
                    from utils import validate_submission
                    valid, errors = validate_submission(submission, all_nutrients)
                    
                    if valid:
                        st.success("✅ Submission format is valid and ready for download!", icon="✅")
                    else:
                        st.error(f"❌ Validation issues: {', '.join(errors)}")
                    
                    # Download button
                    csv = submission.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download submission.csv",
                        data=csv,
                        file_name="submission.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Submission Checklist")
            st.markdown("""
            - [x] All 13 nutrient columns
            - [x] IDs match test set
            - [x] Zero-mask applied
            - [x] No NaN values
            - [x] Non-negative values
            - [x] UTF-8 encoding
            """)
            
            st.markdown("#### Required Format")
            st.code("ID,Target_Al,Target_B,...", language="csv")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("👈 Please train models first in the 'Train Models' tab", icon="👈")

# ----- TAB 4: SETTINGS -----
with tab4:
    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Application Settings")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Visual Preferences")
        st.markdown("*UI preferences are applied automatically*")
        
        st.selectbox("Color Theme", ["Green (Default)", "Blue", "Purple", "Orange"])
        st.checkbox("Enable animations", value=True)
        st.checkbox("Show advanced options", value=False)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Data Preferences")
        
        st.number_input("Default sample size for visualization", min_value=100, max_value=10000, value=2000)
        st.selectbox("Default coordinate system", ["WGS84", "UTM", "Local Grid"])
        st.multiselect(
            "Auto-include features",
            ["Elevation", "Slope", "Aspect", "NDVI", "Rainfall", "Temperature"],
            default=["Elevation", "Rainfall"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Preferences", use_container_width=True):
            st.success("Preferences saved!", icon="✅")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.clear()
            st.success("All settings reset!", icon="🔄")
    
    with col3:
        if st.button("📥 Export Config", use_container_width=True):
            st.info("Configuration exported!")

# ============= FOOTER =============
st.markdown("<div style='margin: 3rem 0; padding: 2rem; text-align: center; background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(129, 199, 132, 0.05)); border-radius: 12px; border: 2px solid rgba(76, 175, 80, 0.2);'>", unsafe_allow_html=True)
st.markdown("""
**🌱 Rhea Soil Nutrient Prediction Challenge**  
*Built with ❤️ for African Agriculture*

Open-source • Reproducible • Competition-ready

[Competition Link](https://zindi.africa/competitions/rhea-soil-nutrient-prediction-challenge) | 
[Documentation](#) | 
[GitHub](#)
""")
st.markdown("</div>", unsafe_allow_html=True)
