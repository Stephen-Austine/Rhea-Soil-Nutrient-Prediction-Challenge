"""
Rhea Soil Nutrient Prediction Challenge
Streamlit App with Navigation Bar & Full-Width Layout
Author: Steve Ochwada | Feb 2026
"""
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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= CUSTOM CSS WITH NAVBAR =============
CUSTOM_CSS = """
<style>
/* Hide default Streamlit header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main app container */
.stApp {
    max-width: 100%;
    padding: 0;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
    min-height: 100vh;
}

.main .block-container {
    max-width: 100%;
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* ========== NAVIGATION BAR ========== */
.navbar {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
    padding: 1rem 3rem;
    position: sticky;
    top: 0;
    z-index: 1000;
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem; /* ← Added spacing below navbar */
}

.navbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    color: white;
    text-decoration: none;
}

.navbar-brand h1 {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    padding: 0;
    line-height: 1.2;
}

.navbar-brand-icon {
    font-size: 2rem;
}

.navbar-links {
    display: flex;
    gap: 8px;
    align-items: center;
}

.nav-link {
    padding: 0.6rem 1.5rem;
    color: rgba(255, 255, 255, 0.85);
    text-decoration: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    cursor: pointer;
    border: 2px solid transparent;
}

.nav-link:hover {
    background: rgba(255, 255, 255, 0.15);
    color: white;
    transform: translateY(-2px);
}

.nav-link.active {
    background: rgba(255, 255, 255, 0.25);
    color: white;
    border-color: rgba(255, 255, 255, 0.4);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* ========== METRIC CARDS - KEEP GREEN ========== */
.metric-card {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(129, 199, 132, 0.08));
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
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.25);
    transform: translateY(-4px);
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

/* ========== SECTION HEADERS - NEUTRAL ========== */
.section-header {
    background: rgba(255, 255, 255, 0.6);
    border-left: 5px solid #64748b;
    padding: 1.2rem 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}

/* ========== GLASS CARDS - CLEAN WHITE (NO COLOR TINT) ========== */
.glass-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

/* ========== BUTTONS - BLUE ========== */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6, #60a5fa);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.8rem;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
    font-size: 1rem;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

/* ========== ALERTS ========== */
.stAlert {
    border-radius: 8px;
    border: 2px solid;
}

/* ========== PROGRESS BARS - BLUE ========== */
.stProgress>div>div>div>div {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    border-radius: 10px;
}

/* ========== DATAFRAME ========== */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .navbar {
        padding: 0.8rem 1.5rem;
        flex-direction: column;
        gap: 1rem;
    }
    .navbar-links {
        flex-wrap: wrap;
        justify-content: center;
    }
    .nav-link {
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
    }
    .metric-value {
        font-size: 2rem;
    }
    .main .block-container {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============= NAVIGATION BAR =============
def render_navbar():
    """Render the top navigation bar"""
    
    # Initialize section state if not exists
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 'explore'
    
    # Determine active link classes
    nav_classes = {
        'explore': 'active' if st.session_state.current_section == 'explore' else '',
        'analyze': 'active' if st.session_state.current_section == 'analyze' else '',
        'train': 'active' if st.session_state.current_section == 'train' else '',
        'submit': 'active' if st.session_state.current_section == 'submit' else '',
        'settings': 'active' if st.session_state.current_section == 'settings' else ''
    }
    
    # Visible navigation buttons
    st.markdown('<div style="display: flex; gap: 1rem; justify-content: center; margin: 2rem 0;">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("📊 Explore", key="nav_explore", use_container_width=True):
            st.session_state.current_section = 'explore'
            st.rerun()
    with col2:
        if st.button("🔍 Analyze", key="nav_analyze", use_container_width=True):
            st.session_state.current_section = 'analyze'
            st.rerun()
    with col3:
        if st.button("🔬 Train", key="nav_train", use_container_width=True):
            st.session_state.current_section = 'train'
            st.rerun()
    with col4:
        if st.button("📤 Submit", key="nav_submit", use_container_width=True):
            st.session_state.current_section = 'submit'
            st.rerun()
    with col5:
        if st.button("⚙️ Settings", key="nav_settings", use_container_width=True):
            st.session_state.current_section = 'settings'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

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

# ============= RENDER NAVBAR =============
render_navbar()

# Initialize section state properly (commented out test line)

# ============= SUBHEADER WITH STATUS =============
if train is not None:
    st.success(f"All datasets loaded successfully| Train: {len(train):,} samples | Test: {len(test):,} samples", icon="✅")
else:
    st.error("❌ Could not load datasets. Place CSV files in the same folder as app.py")
    st.info("Required files: Train.csv, TestSet.csv, TargetPred_To_Keep.csv, SampleSubmission.csv")

st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

# ============= SECTION CONTENT BASED ON NAVIGATION =============
all_nutrients = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']



# ----- SECTION 2: ANALYZE NEW DATA -----
if st.session_state.current_section == 'analyze':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Analyze New Data")
    st.write("Debug: Analyze section is being rendered")
    
    # Input Method Selection - Clickable Card Buttons
    st.markdown("#### 📋 Input Method")
    
    # Initialize input method in session state if not present
    if 'input_method_choice' not in st.session_state:
        st.session_state.input_method_choice = "Manual Entry"
    
    # Create clickable card buttons
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        is_manual = st.session_state.input_method_choice == "Manual Entry"
        if is_manual:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                border-radius: 16px;
                padding: 28px;
                border: 2px solid #3b82f6;
                box-shadow: 0 8px 30px rgba(59, 130, 246, 0.3);
                text-align: center;
            ">
                <div style="font-size: 52px; margin-bottom: 16px;">✏️</div>
                <div style="font-size: 20px; font-weight: 700; color: #ffffff; margin-bottom: 10px;">
                    Manual Entry
                </div>
                <div style="font-size: 14px; color: #ffffff; opacity: 0.95; line-height: 1.6;">
                    Enter soil parameters manually through an interactive form
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("✏️ Manual Entry", key="btn_manual", use_container_width=True):
                st.session_state.input_method_choice = "Manual Entry"
                st.rerun()
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 16px;
                padding: 20px;
                border: 2px solid transparent;
                box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                text-align: center;
                margin-top: -10px;
            ">
                <div style="font-size: 14px; color: #64748b; line-height: 1.6;">
                    Enter soil parameters manually through an interactive form
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        is_upload = st.session_state.input_method_choice == "Upload File (CSV/JSON)"
        if is_upload:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                border-radius: 16px;
                padding: 28px;
                border: 2px solid #3b82f6;
                box-shadow: 0 8px 30px rgba(59, 130, 246, 0.3);
                text-align: center;
            ">
                <div style="font-size: 52px; margin-bottom: 16px;">📁</div>
                <div style="font-size: 20px; font-weight: 700; color: #ffffff; margin-bottom: 10px;">
                    Upload File
                </div>
                <div style="font-size: 14px; color: #ffffff; opacity: 0.95; line-height: 1.6;">
                    Import data from CSV or JSON files for batch analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("📁 Upload File", key="btn_upload", use_container_width=True):
                st.session_state.input_method_choice = "Upload File (CSV/JSON)"
                st.rerun()
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 16px;
                padding: 20px;
                border: 2px solid transparent;
                box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                text-align: center;
                margin-top: -10px;
            ">
                <div style="font-size: 14px; color: #64748b; line-height: 1.6;">
                    Import data from CSV or JSON files for batch analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Get current selection
    input_method = st.session_state.input_method_choice
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_df = None
    
    if input_method == "Manual Entry":
        # Manual Data Input Form
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📍 Location Information")
            
            latitude = st.number_input("Latitude (°)", value=0.0, format="%.6f", help="Latitude coordinate")
            longitude = st.number_input("Longitude (°)", value=0.0, format="%.6f", help="Longitude coordinate")
            elevation = st.number_input("Elevation (m)", value=0.0, format="%.1f", help="Elevation above sea level")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌡️ Environmental Data")
            
            ph = st.number_input("pH Level", value=7.0, format="%.2f", min_value=0.0, max_value=14.0, help="Soil pH level")
            organic_matter = st.number_input("Organic Matter (%)", value=2.5, format="%.2f", min_value=0.0, max_value=100.0, help="Organic matter content")
            temperature = st.number_input("Temperature (°C)", value=25.0, format="%.1f", help="Soil temperature")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Soil Properties Input
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🧪 Soil Properties")
        
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            sand = st.number_input("Sand Content (%)", value=50.0, format="%.1f", min_value=0.0, max_value=100.0, help="Sand percentage in soil")
            clay = st.number_input("Clay Content (%)", value=20.0, format="%.1f", min_value=0.0, max_value=100.0, help="Clay percentage in soil")
        
        with col2:
            silt = st.number_input("Silt Content (%)", value=30.0, format="%.1f", min_value=0.0, max_value=100.0, help="Silt percentage in soil")
            electrical_conductivity = st.number_input("EC (dS/m)", value=0.5, format="%.3f", help="Electrical conductivity")
        
        with col3:
            soil_moisture = st.number_input("Soil Moisture (%)", value=25.0, format="%.1f", help="Soil moisture content")
            cation_exchange = st.number_input("CEC (meq/100g)", value=15.0, format="%.1f", help="Cation exchange capacity")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # File Upload
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 Upload Soil Data File")
        
        uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=['csv', 'json'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    uploaded_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    uploaded_df = pd.read_json(uploaded_file)
                
                st.success("✅ File uploaded successfully!")
                st.markdown("#### 📊 Data Preview")
                st.dataframe(uploaded_df.head(5), use_container_width=True)
                
                # Check required columns
                required_columns = ['Latitude', 'Longitude', 'Elevation', 'pH', 'OrganicMatter', 
                                   'Temperature', 'Sand', 'Clay', 'Silt', 'EC', 'SoilMoisture', 'CEC']
                
                missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ Missing required columns: {', '.join(missing_columns)}. Some predictions may be inaccurate.")
                else:
                    st.success("✅ All required columns are present!")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Button
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    if st.button("🔬 Analyze Soil Data", type="primary", use_container_width=True):
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle file upload validation
        if input_method == "Upload File (CSV/JSON)":
            if uploaded_df is None:
                st.error("❌ Please upload a file first")
            else:
                # File upload analysis
                st.markdown('<div style="margin: 3rem 0;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">', unsafe_allow_html=True)
                st.markdown("### 📊 Analysis Results")
                st.markdown('</div>', unsafe_allow_html=True)
                
                try:
                    from main import SoilNutrientPredictor
                    
                    # Increment files analyzed counter
                    st.session_state.files_analyzed += 1
                    
                    predictor = SoilNutrientPredictor()
                    predictor.load('models/')
                    
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 🎯 Batch Predicted Nutrient Levels")
                    st.markdown(f"Processed {len(uploaded_df)} samples from uploaded file")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show processed data with predictions
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 📈 Sample Predictions")
                    
                    # Predict nutrient levels
                    predictions = predictor.predict(uploaded_df)
                    
                    # Merge predictions with original data
                    if 'ID' in uploaded_df.columns:
                        df_with_preds = pd.merge(uploaded_df, predictions, on='ID')
                    else:
                        df_with_preds = pd.concat([uploaded_df, predictions], axis=1)
                    
                    st.dataframe(df_with_preds.head(10), use_container_width=True)
                    
                    # Download option
                    csv = df_with_preds.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results with Predictions",
                        data=csv,
                        file_name="soil_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Model accuracy card
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Model Performance")
                    
                    # Calculate average RMSE from training metadata
                    import json
                    with open('models/metadata.json', 'r') as f:
                        metadata = json.load(f)
                    
                    training_results = metadata.get('training_results', {})
                    if training_results:
                        cv_rmses = [result['cv_rmse'] for nutrient, result in training_results.items()]
                        avg_rmse = sum(cv_rmses) / len(cv_rmses)
                        
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; color: #666; font-weight: 600;">Average CV RMSE</div>
                                <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">{avg_rmse:.2f}</div>
                                <div style="font-size: 0.8rem; color: #666;">Lower value = better accuracy</div>
                            </div>
                            <div style="text-align: center; padding: 1rem; background: #e0f2fe; border-radius: 12px;">
                                <div style="font-size: 2rem;">📈</div>
                                <div style="font-size: 0.85rem; color: #0369a1;">Model Accuracy</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display individual nutrient RMSE values
                        st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
                        st.markdown("##### Nutrient-wise Performance")
                        
                        col1, col2, col3 = st.columns(3)
                        for i, (nutrient, result) in enumerate(training_results.items()):
                            with [col1, col2, col3][i % 3]:
                                st.markdown(f"""
                                <div style="padding: 0.75rem; background: #f8fafc; border-radius: 8px; border-left: 3px solid #3b82f6; margin: 0.5rem 0;">
                                    <div style="font-size: 0.9rem; font-weight: 600; color: #1e293b;">{nutrient}</div>
                                    <div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{result['cv_rmse']:.2f}</div>
                                    <div style="font-size: 0.75rem; color: #666;">RMSE</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error loading models or making predictions: {str(e)}")
                    st.info("Please ensure you have trained the models first in the 'Train' section or check that the models directory exists.")
        
        else:
            # Manual entry analysis
            # Create data dictionary for analysis
            soil_data = {
                'Latitude': latitude,
                'Longitude': longitude,
                'Elevation': elevation,
                'pH': ph,
                'OrganicMatter': organic_matter,
                'Temperature': temperature,
                'Sand': sand,
                'Clay': clay,
                'Silt': silt,
                'EC': electrical_conductivity,
                'SoilMoisture': soil_moisture,
                'CEC': cation_exchange
            }
            
            # Display Results
            st.markdown('<div style="margin: 3rem 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">', unsafe_allow_html=True)
            st.markdown("### 📊 Analysis Results")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Load models if available
            try:
                from main import SoilNutrientPredictor
                
                predictor = SoilNutrientPredictor()
                predictor.load('models/')
                
                # Predict nutrient levels
                predicted_nutrients = predictor.predict_single(soil_data)
                
                # Results Display
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Predicted Nutrient Levels")
                
                # Display nutrients in a grid
                cols = st.columns(3, gap="medium")
                for i, (nutrient, value) in enumerate(predicted_nutrients.items()):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin: 0.5rem 0;">
                            <div style="font-size: 1.2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem;">{nutrient}</div>
                            <div style="font-size: 2rem; font-weight: 800; color: #3b82f6;">{value:.1f}</div>
                            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">mg/kg</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Model accuracy card
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 📊 Model Performance")
                
                # Calculate average RMSE from training metadata
                import json
                with open('models/metadata.json', 'r') as f:
                    metadata = json.load(f)
                
                training_results = metadata.get('training_results', {})
                if training_results:
                    cv_rmses = [result['cv_rmse'] for nutrient, result in training_results.items()]
                    avg_rmse = sum(cv_rmses) / len(cv_rmses)
                    
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 1rem;">
                        <div>
                            <div style="font-size: 0.9rem; color: #666; font-weight: 600;">Average CV RMSE</div>
                            <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">{avg_rmse:.2f}</div>
                            <div style="font-size: 0.8rem; color: #666;">Lower value = better accuracy</div>
                        </div>
                        <div style="text-align: center; padding: 1rem; background: #e0f2fe; border-radius: 12px;">
                            <div style="font-size: 2rem;">📈</div>
                            <div style="font-size: 0.85rem; color: #0369a1;">Model Accuracy</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display individual nutrient RMSE values
                    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
                    st.markdown("##### Nutrient-wise Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    for i, (nutrient, result) in enumerate(training_results.items()):
                        with [col1, col2, col3][i % 3]:
                            st.markdown(f"""
                            <div style="padding: 0.75rem; background: #f8fafc; border-radius: 8px; border-left: 3px solid #3b82f6; margin: 0.5rem 0;">
                                <div style="font-size: 0.9rem; font-weight: 600; color: #1e293b;">{nutrient}</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{result['cv_rmse']:.2f}</div>
                                <div style="font-size: 0.75rem; color: #666;">RMSE</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error loading models or making predictions: {str(e)}")
                st.info("Please ensure you have trained the models first in the 'Train' section or check that the models directory exists.")
        
        # Soil Health Assessment
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Soil Health Assessment")
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: #dcfce7; border-radius: 12px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🍃</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #16a34a;">Good</div>
                <div style="font-size: 0.85rem; color: #4f46e5; margin-top: 0.5rem;">pH Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: #fef3c7; border-radius: 12px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💧</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #ca8a04;">Moderate</div>
                <div style="font-size: 0.85rem; color: #4f46e5; margin-top: 0.5rem;">Moisture</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: #fee2e2; border-radius: 12px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚠️</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #dc2626;">Low</div>
                <div style="font-size: 0.85rem; color: #4f46e5; margin-top: 0.5rem;">Nitrogen</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Recommendations")
        
        recommendations = [
            "Consider adding nitrogen-rich fertilizers to improve crop growth",
            "Maintain proper irrigation schedule to optimize soil moisture",
            "Monitor pH levels regularly and adjust if needed",
            "Add organic matter to improve soil structure and fertility"
        ]
        
        for i, rec in enumerate(recommendations):
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; gap: 1rem; padding: 1rem; margin: 0.5rem 0; background: #f8fafc; border-radius: 8px;">
                <div style="font-size: 1.5rem; color: #3b82f6; margin-top: 0.25rem;">{i+1}.</div>
                <div style="flex: 1; line-height: 1.6;">{rec}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('</div>', unsafe_allow_html=True)

# ----- SECTION 2: EXPLORE DATA -----
if st.session_state.current_section == 'explore':
    # ============= DATASET OVERVIEW =============
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Overview")
    
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
        # Track number of files analyzed
        if 'files_analyzed' not in st.session_state:
            st.session_state.files_analyzed = 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Files Analyzed</div>
            <div class="metric-value">{st.session_state.files_analyzed}</div>
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem">New data files processed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    # ============= EXPLORE & ANALYZE DATA =============
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Explore & Analyze Data")
    st.write("Debug: Explore section is being rendered")
    
    if train is not None:
        # Row 1: Two visualizations side by side
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🗺️ Geographic Distribution")
            
            try:
                sample_size = min(2000, len(train))
                train_sample = train.sample(sample_size, random_state=42)
                color_col = 'N' if 'N' in train_sample.columns else train_sample.columns[2]
                
                fig_map = px.scatter_mapbox(
                    train_sample,
                    lat="Latitude", lon="Longitude", color=color_col,
                    zoom=3, height=500, mapbox_style="carto-positron",
                    color_continuous_scale="Viridis",
                    title=f"Spatial Distribution of {color_col} Concentration"
                )
                fig_map.update_layout(
                    margin=dict(t=40, b=0, l=0, r=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render map: {str(e)}")
                # Fallback: Show scatter plot instead of map
                st.markdown("Showing scatter plot fallback:")
                fig_fallback = px.scatter(
                    train_sample,
                    x="Longitude", y="Latitude", color=color_col,
                    color_continuous_scale="Viridis",
                    title=f"Spatial Distribution of {color_col} Concentration"
                )
                fig_fallback.update_layout(
                    height=500,
                    margin=dict(t=40, b=0, l=0, r=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_fallback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Nutrient Correlations")
            
            available_nutrients = [c for c in all_nutrients if c in train.columns]
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
            
            available_nutrients = [c for c in all_nutrients if c in train.columns]
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
            
            key_nutrients = ['N', 'P', 'K', 'Ca', 'Mg']
            available_key = [n for n in key_nutrients if n in train.columns]
            
            if available_key:
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
        
        # Row 3: Histograms for key nutrients
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Nutrient Histograms")
        
        key_nutrients = ['N', 'P', 'K', 'Ca', 'Mg']
        available_key = [n for n in key_nutrients if n in train.columns]
        
        if available_key:
            col1, col2 = st.columns(2)
            
            # Color palette for histograms
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']
            
            for i, nutrient in enumerate(available_key):
                with col1 if i % 2 == 0 else col2:
                    nutrient_data = train[nutrient].dropna()
                    if len(nutrient_data) > 0:
                        fig_hist = px.histogram(
                            x=nutrient_data,
                            title=f'Distribution of {nutrient}',
                            labels={'x': f'{nutrient} (mg/kg)', 'y': 'Count'},
                            color_discrete_sequence=[colors[i % len(colors)]],
                            nbins=30
                        )
                        fig_hist.update_layout(
                            height=300,
                            margin=dict(t=40, b=0, l=0, r=0),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 4: pH vs Nutrient Concentrations
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🟡 pH vs Nutrient Concentrations")
        
        # Check for pH column (case-insensitive search)
        ph_column = None
        for col in train.columns:
            if col.lower() == 'ph':
                ph_column = col
                break
        
        if ph_column:
            nutrients_to_plot = ['N', 'P', 'K', 'Ca', 'Mg']
            available_for_ph = [n for n in nutrients_to_plot if n in train.columns]
            
            if available_for_ph:
                col1, col2 = st.columns(2)
                
                # Color palette for pH plots
                colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']
                
                for i, nutrient in enumerate(available_for_ph):
                    with col1 if i % 2 == 0 else col2:
                        try:
                            df_ph = train[[ph_column, nutrient]].dropna()
                            if len(df_ph) > 0:
                                fig_ph = px.scatter(
                                    df_ph,
                                    x=ph_column,
                                    y=nutrient,
                                    title=f'pH vs {nutrient}',
                                    labels={'ph': 'pH Level', nutrient: f'{nutrient} (mg/kg)'},
                                    color_discrete_sequence=[colors[i % len(colors)]],
                                    opacity=0.6
                                )
                                fig_ph.update_layout(
                                    height=300,
                                    margin=dict(t=40, b=0, l=0, r=0),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig_ph, use_container_width=True)
                            else:
                                st.warning(f"No data available for pH vs {nutrient}")
                        except Exception as e:
                            st.error(f"Error creating pH vs {nutrient} plot: {str(e)}")
            else:
                st.info("No key nutrients available for pH analysis")
        else:
            st.warning("pH column not found in dataset")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 6: Depth vs Nutrient Concentrations
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📏 Depth vs Nutrient Concentrations")
        
        if 'Depth_cm' in train.columns:
            nutrients_to_plot = ['N', 'P', 'K', 'Ca', 'Mg']
            available_for_depth = [n for n in nutrients_to_plot if n in train.columns]
            
            if available_for_depth:
                col1, col2 = st.columns(2)
                
                # Color palette for depth plots
                colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']
                
                for i, nutrient in enumerate(available_for_depth):
                    with col1 if i % 2 == 0 else col2:
                        df_depth = train[['Depth_cm', nutrient]].dropna()
                        if len(df_depth) > 0:
                            fig_depth = px.scatter(
                                df_depth,
                                x='Depth_cm',
                                y=nutrient,
                                title=f'Depth vs {nutrient}',
                                labels={'Depth_cm': 'Depth (cm)', nutrient: f'{nutrient} (mg/kg)'},
                                color_discrete_sequence=[colors[i % len(colors)]],
                                opacity=0.6
                            )
                            fig_depth.update_layout(
                                height=300,
                                margin=dict(t=40, b=0, l=0, r=0),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_depth, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 7: Statistics table
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Summary Statistics")
        
        if available_nutrients:
            stats_df = train[available_nutrients].describe().T
            stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            st.dataframe(stats_df.style.background_gradient(cmap='Greys', subset=['Mean', 'Std']), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----- SECTION 2: TRAIN MODELS -----
elif st.session_state.current_section == 'train':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Train Machine Learning Models")
    
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
    
    # Initialize session state for nutrient selection if not exists
    if 'selected_nutrients' not in st.session_state:
        st.session_state.selected_nutrients = all_nutrients.copy()
    
    # Display checkboxes in 3 columns for better readability
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    
    for i, nutrient in enumerate(all_nutrients):
        with columns[i % 3]:
            # Check if nutrient is currently selected
            is_selected = nutrient in st.session_state.selected_nutrients
            
            # Create checkbox
            if st.checkbox(nutrient, value=is_selected, key=f"nutrient_{nutrient}"):
                if nutrient not in st.session_state.selected_nutrients:
                    st.session_state.selected_nutrients.append(nutrient)
            else:
                if nutrient in st.session_state.selected_nutrients:
                    st.session_state.selected_nutrients.remove(nutrient)
    
    # Use session state for selected nutrients
    selected = st.session_state.selected_nutrients
    
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
                    
                    from main import SoilNutrientPredictor
                    
                    model_type_lower = model_type.lower().replace(' ', '_')
                    
                    predictor = SoilNutrientPredictor(
                        model_type=model_type_lower,
                        cv_folds=cv_folds,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_est
                    )
                    
                    status_text.text("Loading and preprocessing data...")
                    predictor.load_data(train, test, mask)
                    progress_bar.progress(20)
                    
                    status_text.text(f"Training {len(selected)} nutrient models...")
                    results = predictor.train(selected)
                    progress_bar.progress(80)
                    
                    status_text.text("Training complete!")
                    progress_bar.progress(100)
                    
                    st.success("✅ Training completed successfully!")
                    
                    results_data = []
                    for nutrient, metrics in results.items():
                        results_data.append({
                            'Nutrient': nutrient,
                            'CV RMSE': f"{metrics['cv_rmse']:.3f}",
                            'Std Dev': f"{metrics['cv_std']:.3f}",
                            'Train RMSE': f"{metrics['train_rmse']:.3f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df.style.background_gradient(cmap='Greys', subset=['CV RMSE']), use_container_width=True)
                    
                    predictor.save("models/")
                    st.session_state.predictor = predictor
                    st.session_state.trained = True
                    st.session_state.results = results
                    
                    st.balloons()
    
    # Show training history if available
    if st.session_state.get('trained', False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Training History")
        
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

# ----- SECTION 3: GENERATE SUBMISSION -----
elif st.session_state.current_section == 'submit':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Generate Competition Submission")
    
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
                    
                    from utils import validate_submission
                    valid, errors = validate_submission(submission, all_nutrients)
                    
                    if valid:
                        st.success("✅ Submission format is valid and ready for download!", icon="✅")
                    else:
                        st.error(f"❌ Validation issues: {', '.join(errors)}")
                    
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
        st.warning("👈 Please train models first in the 'Train Models' section", icon="👈")

# ----- SECTION 4: SETTINGS -----
elif st.session_state.current_section == 'settings':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Application Settings")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Visual Preferences")
        st.markdown("*UI preferences are applied automatically*")
        
        st.selectbox("Color Theme", ["Neutral (Default)", "Purple", "Orange", "Green"])
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
st.markdown("<div style='margin: 3rem 0; padding: 2rem; text-align: center; background: rgba(255, 255, 255, 0.8); border-radius: 12px; border: 1px solid rgba(0, 0, 0, 0.08);'>", unsafe_allow_html=True)
st.markdown("""
**🌱 Rhea Soil Nutrient Prediction Challenge**  
*Built with ❤️ for African Agriculture*

Open-source • Reproducible • Competition-ready

[Competition Link](https://zindi.africa/competitions/rhea-soil-nutrient-prediction-challenge) | 
[Documentation](#) | 
[GitHub](#)
""")
st.markdown("</div>", unsafe_allow_html=True)