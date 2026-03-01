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

# Import for PDF generation
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ============= CROP DATABASE =============
# Comprehensive database of crops with their optimal nutrient requirements
# Format: {crop_name: {nutrient: (min_optimal, max_optimal, weight), ...}}
CROP_DATABASE = {
    # Cereals
    "Maize": {"N": (15, 25, 1.0), "P": (10, 20, 0.9), "K": (15, 25, 0.9), "pH": (5.8, 7.0, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Wheat": {"N": (20, 30, 1.0), "P": (12, 25, 0.9), "K": (15, 30, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (2.5, 5.0, 0.7)},
    "Rice": {"N": (15, 30, 1.0), "P": (8, 18, 0.9), "K": (12, 25, 0.9), "pH": (5.5, 6.5, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    "Sorghum": {"N": (10, 20, 1.0), "P": (8, 15, 0.9), "K": (12, 20, 0.9), "pH": (5.5, 7.5, 0.8), "OrganicMatter": (1.5, 3.5, 0.7)},
    "Barley": {"N": (18, 28, 1.0), "P": (10, 20, 0.9), "K": (15, 25, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Millet": {"N": (8, 15, 1.0), "P": (6, 12, 0.9), "K": (10, 18, 0.9), "pH": (5.5, 7.0, 0.8), "OrganicMatter": (1.0, 2.5, 0.7)},
    
    # Legumes
    "Beans": {"N": (10, 20, 0.8), "P": (15, 25, 1.0), "K": (15, 25, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Peas": {"N": (12, 22, 0.8), "P": (12, 22, 1.0), "K": (12, 22, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (2.5, 4.5, 0.7)},
    "Lentils": {"N": (8, 18, 0.8), "P": (10, 20, 1.0), "K": (12, 20, 0.9), "pH": (6.0, 7.0, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Chickpeas": {"N": (10, 20, 0.8), "P": (12, 22, 1.0), "K": (15, 25, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Soybeans": {"N": (15, 30, 0.8), "P": (15, 30, 1.0), "K": (20, 35, 0.9), "pH": (6.0, 7.0, 0.8), "OrganicMatter": (2.5, 4.5, 0.7)},
    "Groundnuts": {"N": (10, 20, 0.8), "P": (15, 30, 1.0), "K": (20, 35, 0.9), "pH": (5.5, 7.0, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    
    # Root Crops
    "Potatoes": {"N": (20, 35, 1.0), "P": (15, 30, 0.9), "K": (25, 40, 1.0), "pH": (5.0, 6.5, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    "Cassava": {"N": (10, 20, 0.9), "P": (8, 15, 0.8), "K": (15, 25, 0.9), "pH": (5.5, 7.0, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Sweet Potatoes": {"N": (8, 18, 0.9), "P": (12, 22, 0.8), "K": (20, 35, 0.9), "pH": (5.5, 6.5, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Carrots": {"N": (10, 20, 0.9), "P": (15, 30, 0.9), "K": (20, 35, 0.9), "pH": (6.0, 7.0, 0.8), "OrganicMatter": (2.5, 4.5, 0.7)},
    
    # Vegetables
    "Tomatoes": {"N": (20, 40, 1.0), "P": (20, 35, 0.9), "K": (25, 45, 1.0), "pH": (6.0, 6.8, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    "Cabbage": {"N": (25, 45, 1.0), "P": (18, 30, 0.9), "K": (20, 35, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    "Onions": {"N": (15, 30, 0.9), "P": (15, 25, 0.9), "K": (20, 35, 0.9), "pH": (6.0, 7.0, 0.8), "OrganicMatter": (2.5, 4.5, 0.7)},
    "Spinach": {"N": (20, 40, 1.0), "P": (15, 25, 0.9), "K": (20, 35, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    "Kale": {"N": (25, 45, 1.0), "P": (15, 30, 0.9), "K": (20, 35, 0.9), "pH": (6.0, 7.5, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    
    # Cash Crops
    "Coffee": {"N": (15, 25, 0.9), "P": (10, 20, 0.9), "K": (20, 35, 1.0), "pH": (5.5, 6.5, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    "Tea": {"N": (20, 35, 1.0), "P": (12, 22, 0.9), "K": (15, 25, 0.9), "pH": (4.5, 5.5, 0.8), "OrganicMatter": (3.0, 5.0, 0.7)},
    "Cotton": {"N": (20, 35, 1.0), "P": (15, 25, 0.9), "K": (20, 35, 0.9), "pH": (5.5, 7.5, 0.8), "OrganicMatter": (2.0, 4.0, 0.7)},
    "Tobacco": {"N": (25, 45, 1.0), "P": (20, 35, 0.9), "K": (25, 40, 0.9), "pH": (5.5, 6.5, 0.8), "OrganicMatter": (2.5, 4.5, 0.7)},
}

def calculate_crop_suitability(soil_data, predicted_nutrients=None):
    """
    Calculate suitability scores for all crops based on soil conditions.
    
    Args:
        soil_data: Dictionary with soil parameters (pH, OrganicMatter, etc.)
        predicted_nutrients: Dictionary with predicted N, P, K values (optional)
    
    Returns:
        List of tuples (crop_name, suitability_score, matching_nutrients)
    """
    scores = []
    
    for crop_name, requirements in CROP_DATABASE.items():
        total_score = 0
        total_weight = 0
        matching_factors = []
        
        for nutrient, (min_val, max_val, weight) in requirements.items():
            # Get actual value from soil data or predicted nutrients
            if nutrient in soil_data:
                actual_value = soil_data[nutrient]
            elif predicted_nutrients and nutrient in predicted_nutrients:
                actual_value = predicted_nutrients[nutrient]
            else:
                continue
            
            # Calculate suitability score for this nutrient
            if min_val <= actual_value <= max_val:
                # Within optimal range - full score
                nutrient_score = 1.0
                matching_factors.append(f"{nutrient}: Optimal")
            elif actual_value < min_val:
                # Below minimum - partial score based on proximity
                ratio = actual_value / min_val if min_val > 0 else 0
                nutrient_score = max(0, ratio)
                matching_factors.append(f"{nutrient}: Low ({ratio*100:.0f}%)")
            else:
                # Above maximum - partial score based on excess
                ratio = max_val / actual_value if actual_value > 0 else 0
                nutrient_score = max(0, ratio)
                matching_factors.append(f"{nutrient}: High ({ratio*100:.0f}%)")
            
            total_score += nutrient_score * weight
            total_weight += weight
        
        # Calculate final percentage score
        if total_weight > 0:
            final_score = (total_score / total_weight) * 100
        else:
            final_score = 0
        
        scores.append((crop_name, final_score, matching_factors))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def generate_analysis_pdf(soil_data, predictions, crop_scores, input_type="Manual Entry"):
    """
    Generate a PDF report of the soil analysis results.
    
    Args:
        soil_data: Dictionary with soil parameters
        predictions: Dictionary with predicted nutrient levels
        crop_scores: List of crop suitability scores
        input_type: String indicating input method
    
    Returns:
        BytesIO object containing the PDF
    """
    from io import BytesIO
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 15, 'Soil Nutrient Analysis Report', 0, 1, 'C')
    pdf.ln(5)
    
    # Date and Input Type
    pdf.set_font('Arial', '', 12)
    from datetime import datetime
    pdf.cell(0, 10, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'L')
    pdf.cell(0, 10, f'Analysis Type: {input_type}', 0, 1, 'L')
    pdf.ln(5)
    
    # Soil Input Data Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Soil Input Parameters', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    for key, value in soil_data.items():
        if isinstance(value, (int, float)):
            pdf.cell(0, 8, f'{key}: {value:.2f}', 0, 1, 'L')
        else:
            pdf.cell(0, 8, f'{key}: {value}', 0, 1, 'L')
    pdf.ln(5)
    
    # Predicted Nutrients Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Predicted Nutrient Levels', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    if predictions:
        for nutrient, value in predictions.items():
            pdf.cell(0, 8, f'{nutrient}: {value:.2f} mg/kg', 0, 1, 'L')
    pdf.ln(5)
    
    # Recommended Crops Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommended Crops (Top 5)', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    for i, (crop_name, score, _) in enumerate(crop_scores[:5], 1):
        pdf.cell(0, 8, f'{i}. {crop_name}: {score:.1f}% suitability', 0, 1, 'L')
    pdf.ln(5)
    
    # Not Recommended Crops Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Not Recommended Crops (Bottom 5)', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    for i, (crop_name, score, _) in enumerate(crop_scores[-5:], 1):
        pdf.cell(0, 8, f'{i}. {crop_name}: {score:.1f}% suitability', 0, 1, 'L')
    pdf.ln(10)
    
    # Footer
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, 'Generated by Rhea Soil Nutrient Predictor', 0, 1, 'C')
    
    # Output to BytesIO
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output


# ============= PAGE CONFIG - MUST BE FIRST =============
st.set_page_config(
    page_title="Rhea Soil Nutrient Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= INITIALIZE SESSION STATE =============
if 'analyses_count' not in st.session_state:
    st.session_state.analyses_count = 0

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

# ============= DARK MODE CSS =============
DARK_MODE_CSS = """
<style>
/* ===== DARK MODE - ROOT ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
}

/* ===== TEXT COLORS ===== */
.stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label {
    color: #f1f5f9 !important;
}

/* Subtitle text */
.stMarkdown p {
    color: #cbd5e1 !important;
}

/* ===== CARDS ===== */
.glass-card {
    background: rgba(30, 41, 59, 0.95) !important;
    border: 1px solid rgba(71, 85, 105, 0.5) !important;
}

.metric-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05)) !important;
    border: 2px solid rgba(34, 197, 94, 0.3) !important;
}

.metric-value { color: #4ade80 !important; }
.metric-label { color: #86efac !important; }

/* ===== FORM ELEMENTS ===== */
/* Selectbox, NumberInput containers */
.stSelectbox > div > div,
.stNumberInput > div > div {
    background-color: #1e293b !important;
    border-color: #475569 !important;
}

/* Input text */
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {
    color: #f1f5f9 !important;
}

/* Slider */
.stSlider > div > div { background-color: #334155 !important; }
.stSlider > div > div > div > div { background-color: #3b82f6 !important; }

/* Checkbox label */
.stCheckbox > label { color: #f1f5f9 !important; }

/* Toggle switch */
[data-testid="stToggle"] > div > div { background-color: #334155 !important; }
[data-testid="stToggle"][aria-checked="true"] > div > div { background-color: #3b82f6 !important; }

/* ===== BUTTONS ===== */
.stButton > button {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
}

.stButton > button:hover { background-color: #2563eb !important; }

/* ===== ALERTS ===== */
.stAlert { background-color: rgba(30, 41, 59, 0.9) !important; }

[data-testid="stAlertContentSuccess"] {
    background-color: rgba(22, 163, 74, 0.2) !important;
    border-left-color: #22c55e !important;
}

[data-testid="stAlertContentError"] {
    background-color: rgba(239, 68, 68, 0.2) !important;
    border-left-color: #ef4444 !important;
}

[data-testid="stAlertContentInfo"] {
    background-color: rgba(59, 130, 246, 0.2) !important;
    border-left-color: #3b82f6 !important;
}

.stAlert [data-testid="stMarkdownContainer"] p { color: #f1f5f9 !important; }

/* ===== DATAFRAMES & TABLES ===== */
[data-testid="stDataFrame"] { background-color: #1e293b !important; }
[data-testid="stDataFrame"] th {
    background-color: #334155 !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}
[data-testid="stDataFrame"] td {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: #475569 !important;
}

/* ===== NAVIGATION ===== */
.navbar {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%) !important;
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
}

.streamlit-expanderContent {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background-color: #1e293b !important;
    border-bottom-color: #475569 !important;
}

.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
}

/* ===== FILE UPLOADER ===== */
.stFileUploader > div > div {
    background-color: #1e293b !important;
    border-color: #475569 !important;
}

/* ===== PLOTLY CHARTS - TEXT VISIBILITY ===== */
/* Make plot text white/light in dark mode so it's visible on dark backgrounds */
.js-plotly-plot .plotly text,
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle,
.js-plotly-plot .plotly .legendtext,
.js-plotly-plot .plotly .g-xtitle,
.js-plotly-plot .plotly .g-ytitle,
.js-plotly-plot .plotly .axis-title,
.js-plotly-plot .plotly .scatterlayer .trace text {
    fill: #f1f5f9 !important;
    color: #f1f5f9 !important;
}

/* Axis labels and tick text */
.js-plotly-plot .plotly .g-xticklabel,
.js-plotly-plot .plotly .g-yticklabel,
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text {
    fill: #cbd5e1 !important;
}

/* Hover labels */
.js-plotly-plot .plotly .hovertext {
    fill: #0f172a !important;
}

/* Legend background */
.js-plotly-plot .plotly .legend {
    background-color: rgba(30, 41, 59, 0.9) !important;
}

/* ===== PROGRESS BARS ===== */
.stProgress > div > div { background-color: #334155 !important; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
}

/* ===== LINKS ===== */
a { color: #60a5fa !important; }
a:hover { color: #93c5fd !important; }

/* ===== TOOLTIPS ===== */
[data-testid="stTooltip"] {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #475569 !important;
}

/* ===== METRICS ===== */
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricDelta"] { color: #22c55e !important; }

/* ===== CODE BLOCKS ===== */
.stCodeBlock { background-color: #1e293b !important; }
.stCodeBlock code { color: #f1f5f9 !important; }

/* ===== FOOTER ===== */
footer { background: rgba(30, 41, 59, 0.8) !important; }
footer p { color: #94a3b8 !important; }

/* ===== HORIZONTAL RULE ===== */
hr { border-color: #475569 !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { background-color: #1e293b !important; }
::-webkit-scrollbar-thumb { background-color: #475569 !important; }
</style>
"""

# ============= APPLY CSS BASED ON DARK MODE SETTING =============
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    st.markdown(CUSTOM_CSS + DARK_MODE_CSS, unsafe_allow_html=True)
else:
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
        'simple': 'active' if st.session_state.current_section == 'simple' else '',
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
        if st.button("Advanced Analysis", key="nav_analyze", use_container_width=True):
            st.session_state.current_section = 'analyze'
            st.rerun()
    with col3:
        if st.button("🔬 Train", key="nav_train", use_container_width=True):
            st.session_state.current_section = 'train'
            st.rerun()
    with col4:
        if st.button("🌾 Analysis", key="nav_simple", use_container_width=True):
            st.session_state.current_section = 'simple'
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
    st.markdown("### Advanced Analysis")
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
                    
                    # Increment analyses counter
                    st.session_state.analyses_count += 1
                    
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
                    
                    # ============= CROP RECOMMENDATIONS FOR BATCH =============
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 🌾 Crop Recommendations (Batch Average)")
                    st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Based on average soil conditions across all uploaded samples.</p>", unsafe_allow_html=True)
                    
                    # Calculate average nutrient values from predictions for crop recommendations
                    avg_soil_data = {
                        'pH': uploaded_df['pH'].mean() if 'pH' in uploaded_df.columns else 6.5,
                        'OrganicMatter': uploaded_df['OrganicMatter'].mean() if 'OrganicMatter' in uploaded_df.columns else 3.0,
                    }
                    
                    # Add predicted nutrient averages
                    nutrient_cols = ['N', 'P', 'K']
                    for col in nutrient_cols:
                        if col in predictions.columns:
                            avg_soil_data[col] = predictions[col].mean()
                        elif col in uploaded_df.columns:
                            avg_soil_data[col] = uploaded_df[col].mean()
                        else:
                            avg_soil_data[col] = 15.0  # Default value
                    
                    # Calculate crop suitability
                    batch_crop_scores = calculate_crop_suitability(avg_soil_data)
                    
                    # Create two columns
                    col_rec, col_not_rec = st.columns(2, gap="large")
                    
                    with col_rec:
                        st.markdown("<div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #16a34a;'>", unsafe_allow_html=True)
                        st.markdown("<h5 style='color: #16a34a; margin-bottom: 1rem;'>✅ Recommended Crops</h5>", unsafe_allow_html=True)
                        st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 1rem;'>Top 5 crops best suited for average soil conditions</p>", unsafe_allow_html=True)
                        
                        for i, (crop_name, score, factors) in enumerate(batch_crop_scores[:5]):
                            score_color = "#16a34a" if score >= 70 else "#ca8a04" if score >= 50 else "#dc2626"
                            score_bg = "#f0fdf4" if score >= 70 else "#fefce8" if score >= 50 else "#fef2f2"
                            
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; margin: 0.5rem 0; background: {score_bg}; border-radius: 8px; border: 1px solid #e2e8f0;">
                                <div style="display: flex; align-items: center; gap: 0.75rem;">
                                    <span style="font-size: 1.25rem;">{['🌽', '🌾', '🌱', '🫘', '🥬'][i]}</span>
                                    <span style="font-weight: 600; color: #1e293b;">{crop_name}</span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.25rem; font-weight: 700; color: {score_color};">{score:.0f}%</div>
                                    <div style="font-size: 0.7rem; color: #666;">Suitability</div>
                                </div>
                            </div>
                            <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: -0.25rem; margin-bottom: 0.5rem;">
                                <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {score_color} 0%, {score_color}80 100%); border-radius: 2px;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_not_rec:
                        st.markdown("<div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #dc2626;'>", unsafe_allow_html=True)
                        st.markdown("<h5 style='color: #dc2626; margin-bottom: 1rem;'>❌ Not Recommended</h5>", unsafe_allow_html=True)
                        st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 1rem;'>Crops that may struggle in current soil conditions</p>", unsafe_allow_html=True)
                        
                        for i, (crop_name, score, factors) in enumerate(batch_crop_scores[-5:]):
                            score_color = "#16a34a" if score >= 70 else "#ca8a04" if score >= 50 else "#dc2626"
                            score_bg = "#fef2f2" if score < 40 else "#fefce8" if score < 60 else "#f0fdf4"
                            
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; margin: 0.5rem 0; background: {score_bg}; border-radius: 8px; border: 1px solid #e2e8f0;">
                                <div style="display: flex; align-items: center; gap: 0.75rem;">
                                    <span style="font-size: 1.25rem;">{['🍅', '🥔', '🧅', '🍵', '🌿'][i]}</span>
                                    <span style="font-weight: 600; color: #1e293b;">{crop_name}</span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.25rem; font-weight: 700; color: {score_color};">{score:.0f}%</div>
                                    <div style="font-size: 0.7rem; color: #666;">Suitability</div>
                                </div>
                            </div>
                            <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: -0.25rem; margin-bottom: 0.5rem;">
                                <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {score_color} 0%, {score_color}80 100%); border-radius: 2px;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ============= DOWNLOAD BATCH ANALYSIS RESULTS =============
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 💾 Download Batch Analysis Results")
                    st.markdown(f"<p style='color: #666; margin-bottom: 1.5rem;'>Export results for {len(df_with_preds)} analyzed samples.</p>", unsafe_allow_html=True)
                    
                    col_csv, col_pdf = st.columns(2, gap="large")
                    
                    with col_csv:
                        # CSV download for full batch results
                        csv_batch = df_with_preds.to_csv(index=False)
                        
                        st.download_button(
                            label="📄 Download Full Results (CSV)",
                            data=csv_batch,
                            file_name=f"batch_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv_batch"
                        )
                        st.markdown("<p style='font-size: 0.8rem; color: #666; text-align: center;'>Complete dataset with all predictions</p>", unsafe_allow_html=True)
                    
                    with col_pdf:
                        if FPDF_AVAILABLE:
                            # Generate PDF for batch summary
                            pdf_bytes = generate_analysis_pdf(avg_soil_data, avg_soil_data, batch_crop_scores, f"Batch Upload ({len(df_with_preds)} samples)")
                            
                            st.download_button(
                                label="📑 Download Summary Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"batch_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="download_pdf_batch"
                            )
                            st.markdown("<p style='font-size: 0.8rem; color: #666; text-align: center;'>Summary report with averages</p>", unsafe_allow_html=True)
                        else:
                            st.info("📦 Install fpdf2 to enable PDF downloads")
                    
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
                
                # Increment analyses counter
                st.session_state.analyses_count += 1
                
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
        
        # ============= CROP RECOMMENDATIONS =============
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🌾 Crop Recommendations")
        st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Based on your soil analysis, here are the most suitable and unsuitable crops for your farm.</p>", unsafe_allow_html=True)
        
        # Calculate crop suitability scores
        crop_scores = calculate_crop_suitability(soil_data, predicted_nutrients)
        
        # Create two columns for recommended and not recommended
        col_rec, col_not_rec = st.columns(2, gap="large")
        
        with col_rec:
            st.markdown("<div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #16a34a;'>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #16a34a; margin-bottom: 1rem;'>✅ Recommended Crops</h5>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 1rem;'>Top 5 crops best suited for your soil conditions</p>", unsafe_allow_html=True)
            
            # Top 5 recommended crops
            for i, (crop_name, score, factors) in enumerate(crop_scores[:5]):
                score_color = "#16a34a" if score >= 70 else "#ca8a04" if score >= 50 else "#dc2626"
                score_bg = "#f0fdf4" if score >= 70 else "#fefce8" if score >= 50 else "#fef2f2"
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; margin: 0.5rem 0; background: {score_bg}; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.25rem;">{['🌽', '🌾', '🌱', '🫘', '🥬'][i]}</span>
                        <span style="font-weight: 600; color: #1e293b;">{crop_name}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.25rem; font-weight: 700; color: {score_color};">{score:.0f}%</div>
                        <div style="font-size: 0.7rem; color: #666;">Suitability</div>
                    </div>
                </div>
                <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: -0.25rem; margin-bottom: 0.5rem;">
                    <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {score_color} 0%, {score_color}80 100%); border-radius: 2px; transition: width 0.5s ease;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_not_rec:
            st.markdown("<div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #dc2626;'>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #dc2626; margin-bottom: 1rem;'>❌ Not Recommended</h5>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 1rem;'>Crops that may struggle in your current soil conditions</p>", unsafe_allow_html=True)
            
            # Bottom 5 not recommended crops
            for i, (crop_name, score, factors) in enumerate(crop_scores[-5:]):
                score_color = "#16a34a" if score >= 70 else "#ca8a04" if score >= 50 else "#dc2626"
                score_bg = "#fef2f2" if score < 40 else "#fefce8" if score < 60 else "#f0fdf4"
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; margin: 0.5rem 0; background: {score_bg}; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.25rem;">{['🍅', '🥔', '🧅', '🍵', '🌿'][i]}</span>
                        <span style="font-weight: 600; color: #1e293b;">{crop_name}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.25rem; font-weight: 700; color: {score_color};">{score:.0f}%</div>
                        <div style="font-size: 0.7rem; color: #666;">Suitability</div>
                    </div>
                </div>
                <div style="width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: -0.25rem; margin-bottom: 0.5rem;">
                    <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {score_color} 0%, {score_color}80 100%); border-radius: 2px; transition: width 0.5s ease;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Add legend/explanation
        st.markdown("<div style='margin-top: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 8px;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.85rem; color: #666; margin: 0;'><strong>📊 How to read:</strong> Suitability scores are calculated based on optimal nutrient requirements for each crop. Scores above 70% indicate excellent conditions, 50-70% are acceptable, and below 50% suggest the crop may require soil amendments.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ============= DOWNLOAD ANALYSIS RESULTS =============
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💾 Download Analysis Results")
        st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Export your soil analysis results for record keeping or sharing.</p>", unsafe_allow_html=True)
        
        col_csv, col_pdf = st.columns(2, gap="large")
        
        with col_csv:
            # Prepare CSV data
            csv_results = {
                'Parameter': [],
                'Value': [],
                'Unit': []
            }
            
            # Add soil input data
            for key, value in soil_data.items():
                csv_results['Parameter'].append(key)
                csv_results['Parameter'].append(value if isinstance(value, str) else f"{value:.2f}")
                csv_results['Unit'].append('')
            
            # Add predictions
            if predicted_nutrients:
                for nutrient, value in predicted_nutrients.items():
                    csv_results['Parameter'].append(f"Predicted {nutrient}")
                    csv_results['Value'].append(f"{value:.2f}")
                    csv_results['Unit'].append('mg/kg')
            
            # Add crop recommendations
            csv_results['Parameter'].append('---')
            csv_results['Value'].append('---')
            csv_results['Unit'].append('---')
            
            csv_results['Parameter'].append('RECOMMENDED CROPS')
            csv_results['Value'].append('')
            csv_results['Unit'].append('')
            
            for i, (crop_name, score, _) in enumerate(crop_scores[:5], 1):
                csv_results['Parameter'].append(f"{i}. {crop_name}")
                csv_results['Value'].append(f"{score:.1f}")
                csv_results['Unit'].append('%')
            
            csv_results['Parameter'].append('NOT RECOMMENDED CROPS')
            csv_results['Value'].append('')
            csv_results['Unit'].append('')
            
            for i, (crop_name, score, _) in enumerate(crop_scores[-5:], 1):
                csv_results['Parameter'].append(f"{i}. {crop_name}")
                csv_results['Value'].append(f"{score:.1f}")
                csv_results['Unit'].append('%')
            
            # Ensure all lists have the same length
            max_len = max(len(csv_results['Parameter']), len(csv_results['Value']), len(csv_results['Unit']))
            for key in csv_results:
                while len(csv_results[key]) < max_len:
                    csv_results[key].append('')
            
            df_results = pd.DataFrame(csv_results)
            csv_data = df_results.to_csv(index=False)
            
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name=f"soil_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_manual"
            )
            st.markdown("<p style='font-size: 0.8rem; color: #666; text-align: center;'>Spreadsheet format compatible with Excel</p>", unsafe_allow_html=True)
        
        with col_pdf:
            if FPDF_AVAILABLE:
                # Generate PDF
                pdf_bytes = generate_analysis_pdf(soil_data, predicted_nutrients, crop_scores, "Manual Entry")
                
                st.download_button(
                    label="📑 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"soil_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_manual"
                )
                st.markdown("<p style='font-size: 0.8rem; color: #666; text-align: center;'>Formatted PDF report for sharing</p>", unsafe_allow_html=True)
            else:
                st.info("📦 Install fpdf2 to enable PDF downloads: `pip install fpdf2`")
        
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
        # Display analyses counter
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Analyses Done</div>
            <div class="metric-value">{st.session_state.analyses_count}</div>
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem">Total analyses performed</div>
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
            ["XGBoost", "LightGBM", "Random Forest", "Ridge", "Lasso", "ElasticNet"],
            index=0,
            help="XGBoost recommended for best performance. Ridge/Lasso/ElasticNet are linear models with regularization."
        )
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Hyperparameters")
        
        # Show different parameters based on model type
        if model_type in ["Ridge", "Lasso", "ElasticNet"]:
            alpha = st.slider("Regularization (Alpha)", 0.01, 10.0, 1.0, step=0.01)
            if model_type == "ElasticNet":
                l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, step=0.05)
            else:
                l1_ratio = 0.5
            # Set tree-based params to None for linear models
            max_depth = 8
            learning_rate = 0.1
            n_est = 200
        else:
            max_depth = st.slider("Max Tree Depth", 3, 15, 8)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
            alpha = 1.0
            l1_ratio = 0.5
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Training Options")
        # Only show n_estimators for tree-based models
        if model_type not in ["Ridge", "Lasso", "ElasticNet"]:
            n_est = st.slider("Number of Estimators", 50, 500, 200, step=50)
        else:
            n_est = 200
            st.markdown("<p style='color: #666; font-size: 0.85rem;'>ℹ️ Linear models don't use estimators parameter</p>", unsafe_allow_html=True)
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
                        n_estimators=n_est,
                        alpha=alpha,
                        l1_ratio=l1_ratio
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

# ----- SECTION 3: SIMPLE ANALYSIS -----
elif st.session_state.current_section == 'simple':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🌾 Location-Based Analysis")
    st.markdown("<p style='color: #666;'>Get personalized crop recommendations for your farm location. Simple and easy to use!</p>", unsafe_allow_html=True)
    
    # Initialize simple analysis session state
    if 'simple_analysis_results' not in st.session_state:
        st.session_state.simple_analysis_results = None
    if 'simple_location' not in st.session_state:
        st.session_state.simple_location = {'lat': None, 'lon': None}
    
    # === LOCATION INFORMATION SECTION ===
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📍 Location Information")
    st.markdown("<p style='color: #666; font-size: 0.9rem;'>Enter your farm's coordinates to get personalized recommendations.</p>", unsafe_allow_html=True)
    
    loc_col1, loc_col2 = st.columns(2)
    
    with loc_col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=st.session_state.simple_location['lat'] if st.session_state.simple_location['lat'] else 0.0,
            format="%.6f",
            help="Enter latitude (e.g., -1.2921 for Nairobi)"
        )
    
    with loc_col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.simple_location['lon'] if st.session_state.simple_location['lon'] else 0.0,
            format="%.6f",
            help="Enter longitude (e.g., 36.8219 for Nairobi)"
        )
    
    # === ANALYZE BUTTON ===
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    
    with analyze_col2:
        analyze_clicked = st.button("🔍 Analyze Location", type="primary", use_container_width=True)
    
    if analyze_clicked:
        with st.spinner("Checking location availability..."):
            try:
                # Store location in session state
                st.session_state.simple_location = {'lat': latitude, 'lon': longitude}
                
                # Find nearest soil data from training set based on coordinates
                if train is not None:
                    # Calculate distances to all training points
                    train['distance'] = np.sqrt(
                        (train['Latitude'] - latitude) ** 2 + 
                        (train['Longitude'] - longitude) ** 2
                    )
                    
                    # Get the nearest sample and its distance
                    nearest_idx = train['distance'].idxmin()
                    nearest_distance_deg = train.loc[nearest_idx, 'distance']
                    nearest_distance_km = nearest_distance_deg * 111  # Rough conversion to km
                    
                    # Define maximum acceptable distance (50 km)
                    MAX_DISTANCE_KM = 50
                    
                    if nearest_distance_km > MAX_DISTANCE_KM:
                        # Location is outside available data region
                        st.session_state.simple_analysis_results = {
                            'error': True,
                            'message': f"📍 **Location Data Not Available**\n\nThe coordinates you entered ({latitude:.4f}°, {longitude:.4f}°) are outside our current data coverage area.\n\n**Nearest data point:** {nearest_distance_km:.1f} km away\n**Coverage radius:** {MAX_DISTANCE_KM} km\n\nPlease try a location closer to our monitored regions, or use the **Advanced Analysis** page to manually input soil data.",
                            'nearest_distance_km': nearest_distance_km
                        }
                    else:
                        # Location is within available region - proceed with analysis
                        nearest_sample = train.loc[nearest_idx]
                        
                        # Build soil data dictionary for crop suitability
                        soil_data = {
                            'Latitude': latitude,
                            'Longitude': longitude,
                            'distance_km': nearest_distance_km
                        }
                        
                        # Add predicted nutrients from nearest sample (target columns)
                        target_cols = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Al', 'Na']
                        predicted_nutrients = {}
                        for nutrient in target_cols:
                            if nutrient in nearest_sample:
                                predicted_nutrients[nutrient] = nearest_sample[nutrient]
                                soil_data[nutrient] = nearest_sample[nutrient]
                        
                        # Calculate crop suitability
                        crop_scores = calculate_crop_suitability(soil_data, predicted_nutrients)
                        
                        # Store results
                        st.session_state.simple_analysis_results = {
                            'soil_data': soil_data,
                            'predicted_nutrients': predicted_nutrients,
                            'crop_scores': crop_scores,
                            'nearest_distance_km': nearest_distance_km,
                            'error': False
                        }
                        
                        st.success(f"✅ Analysis complete! Using soil data from {nearest_distance_km:.1f} km away.")
                else:
                    # No training data available
                    st.session_state.simple_analysis_results = {
                        'error': True,
                        'message': "📍 **Location Data Not Available**\n\nOur soil database is currently unavailable. Please try again later, or use the **Advanced Analysis** page to manually input your soil data."
                    }
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.session_state.simple_analysis_results = {'error': True, 'message': f"An error occurred: {str(e)}"}
    
    # === RECOMMENDATIONS SECTION ===
    if st.session_state.simple_analysis_results is not None:
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        results = st.session_state.simple_analysis_results
        
        # Check if there's an error (location outside available region)
        if results.get('error', False):
            st.markdown("### ❌ Location Not Available")
            st.markdown(f"""
                <div style="padding: 2.5rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                            border-radius: 12px; border-left: 4px solid #f59e0b; text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1.5rem;">📍</div>
                    <div style="color: #92400e; line-height: 1.8; font-size: 1.15rem;">
                        {results['message']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
            
            # Buttons for user options
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                # Option to try again with different location
                if st.button("🔄 Try Different Location", use_container_width=True):
                    st.session_state.simple_analysis_results = None
                    st.rerun()
            
            with btn_col2:
                # Button to navigate to Advanced Analysis
                if st.button("🔬 Go to Advanced Analysis →", type="primary", use_container_width=True):
                    st.session_state.simple_analysis_results = None
                    st.session_state.current_section = 'analyze'
                    st.rerun()
        
        else:
            # Location is valid - show recommendations
            st.markdown("### 📊 Recommendations")
            
            crop_scores = results['crop_scores']
            soil_data = results['soil_data']
            predicted_nutrients = results['predicted_nutrients']
            
            # Display location info
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**📍 Coordinates:** {soil_data['Latitude']:.4f}°, {soil_data['Longitude']:.4f}°")
            with info_col2:
                if results.get('nearest_distance_km'):
                    st.markdown(f"**📏 Data Source:** {results['nearest_distance_km']:.1f} km from your location")
            
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
            
            # Create two columns for recommendations
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown('<div style="padding: 1rem; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 12px; border-left: 4px solid #16a34a;">', unsafe_allow_html=True)
                st.markdown("<h5 style='color: #16a34a; margin-bottom: 1rem;'>✅ Top 5 Suitable Crops</h5>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 1rem;'>These crops are best suited for your soil conditions</p>", unsafe_allow_html=True)
                
                for i, (crop_name, score, factors) in enumerate(crop_scores[:5]):
                    score_color = "#16a34a" if score >= 70 else "#ca8a04" if score >= 50 else "#dc2626"
                    st.markdown(f"""
                        <div style="margin-bottom: 0.75rem; padding: 0.75rem; background: rgba(255,255,255,0.8); border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 600; color: #1e293b;">{['🌽', '🌾', '🌱', '🫘', '🥬'][i]} {crop_name}</span>
                                <span style="font-weight: 700; color: {score_color};">{score:.0f}%</span>
                            </div>
                            <div style="margin-top: 0.25rem; height: 4px; background: #e2e8f0; border-radius: 2px;">
                                <div style="width: {score}%; height: 100%; background: {score_color}; border-radius: 2px;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown('<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 12px; border-left: 4px solid #dc2626;">', unsafe_allow_html=True)
                st.markdown("<h5 style='color: #dc2626; margin-bottom: 1rem;'>❌ Top 5 Unsuitable Crops</h5>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 1rem;'>These crops may not thrive in your soil conditions</p>", unsafe_allow_html=True)
                
                for i, (crop_name, score, factors) in enumerate(crop_scores[-5:]):
                    score_color = "#dc2626"
                    st.markdown(f"""
                        <div style="margin-bottom: 0.75rem; padding: 0.75rem; background: rgba(255,255,255,0.8); border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 600; color: #1e293b;">{['🍅', '🥔', '🧅', '🍵', '🌿'][i]} {crop_name}</span>
                                <span style="font-weight: 700; color: {score_color};">{score:.0f}%</span>
                            </div>
                            <div style="margin-top: 0.25rem; height: 4px; background: #e2e8f0; border-radius: 2px;">
                                <div style="width: {score}%; height: 100%; background: {score_color}; border-radius: 2px;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # === DOWNLOAD BUTTONS ===
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### 💾 Download Results")
            
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                # Generate PDF
                try:
                    pdf_bytes = generate_analysis_pdf(soil_data, predicted_nutrients, crop_scores, f"Location Analysis ({latitude:.4f}, {longitude:.4f})")
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"soil_analysis_{latitude:.4f}_{longitude:.4f}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
            
            with dl_col2:
                # Generate CSV
                csv_results = {
                    'Parameter': ['Latitude', 'Longitude'],
                    'Value': [latitude, longitude]
                }
                
                # Add nutrients
                for nutrient, value in predicted_nutrients.items():
                    csv_results['Parameter'].append(f'{nutrient} (Predicted)')
                    csv_results['Value'].append(f'{value:.2f}')
                
                # Add top 5 suitable crops
                csv_results['Parameter'].append('---')
                csv_results['Value'].append('---')
                csv_results['Parameter'].append('TOP 5 SUITABLE CROPS')
                csv_results['Value'].append('')
                
                for i, (crop_name, score, _) in enumerate(crop_scores[:5], 1):
                    csv_results['Parameter'].append(f"{i}. {crop_name}")
                    csv_results['Value'].append(f"{score:.1f}%")
                
                # Add top 5 unsuitable crops
                csv_results['Parameter'].append('---')
                csv_results['Value'].append('---')
                csv_results['Parameter'].append('TOP 5 UNSUITABLE CROPS')
                csv_results['Value'].append('')
                
                for i, (crop_name, score, _) in enumerate(crop_scores[-5:], 1):
                    csv_results['Parameter'].append(f"{i}. {crop_name}")
                    csv_results['Value'].append(f"{score:.1f}%")
                
                results_df = pd.DataFrame(csv_results)
                csv_data = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"soil_analysis_{latitude:.4f}_{longitude:.4f}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with dl_col3:
                # New Analysis button
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.simple_analysis_results = None
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----- SECTION 4: BATCH EXPORT (DEPRECATED - MOVED TO ADVANCED ANALYSIS) -----
elif st.session_state.current_section == 'submit':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Batch Export")
    st.info("🔄 **Batch Export functionality has been integrated into the Advanced Analysis page.**")
    st.markdown("""
    <p style='color: #666;'>Please use the <strong>Advanced Analysis</strong> page to:</p>
    <ul style='color: #666;'>
        <li>Upload batch datasets for analysis</li>
        <li>Export predictions in multiple formats (CSV, Excel, PDF)</li>
        <li>Generate comprehensive reports</li>
    </ul>
    """, unsafe_allow_html=True)
    
    if st.button("Go to Advanced Analysis →", type="primary"):
        st.session_state.current_section = 'analyze'
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----- SECTION 4: SETTINGS -----
elif st.session_state.current_section == 'settings':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    st.markdown("<p style='color: #666;'>Customize your soil analysis experience</p>", unsafe_allow_html=True)
    
    # Initialize settings in session state if not present
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'dark_mode': False,
            'crop_recommendations_count': 5,
            'suitability_threshold': 50,
            'export_format': 'Both CSV & PDF',
            'default_input_method': 'Manual Entry',
            'show_crop_icons': True,
            'show_progress_bars': True,
        }
    
    # Sync dark_mode between session_state.settings and session_state.dark_mode
    if st.session_state.settings.get('dark_mode', False) != st.session_state.dark_mode:
        st.session_state.dark_mode = st.session_state.settings.get('dark_mode', False)
    
    # Create two columns for layout
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Visual Preferences
        st.markdown("#### 🎨 Appearance")
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.settings.get('dark_mode', False), help="Toggle between light and dark theme")
        if dark_mode != st.session_state.settings.get('dark_mode', False):
            st.session_state.settings['dark_mode'] = dark_mode
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analysis Preferences
        st.markdown("#### 🔧 Analysis Preferences")
        default_input = st.selectbox(
            "Default Input Method",
            ["Manual Entry", "File Upload (CSV/JSON)"],
            index=["Manual Entry", "File Upload (CSV/JSON)"].index(st.session_state.settings.get('default_input_method', 'Manual Entry'))
        )
        st.session_state.settings['default_input_method'] = default_input
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Export Settings
        st.markdown("#### 💾 Export Format")
        export_format = st.selectbox(
            "Default Export Format",
            ["Both CSV & PDF", "CSV Only", "PDF Only"],
            index=["Both CSV & PDF", "CSV Only", "PDF Only"].index(st.session_state.settings.get('export_format', 'Both CSV & PDF'))
        )
        st.session_state.settings['export_format'] = export_format
    
    with col2:
        # Crop Recommendations
        st.markdown("#### 🌾 Crop Recommendations")
        crop_count = st.number_input(
            "Number of crops to display",
            min_value=3,
            max_value=10,
            value=st.session_state.settings.get('crop_recommendations_count', 5),
            step=1
        )
        st.session_state.settings['crop_recommendations_count'] = crop_count
        
        threshold = st.slider(
            "Minimum suitability threshold (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.settings.get('suitability_threshold', 50),
            step=5
        )
        st.session_state.settings['suitability_threshold'] = threshold
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Options
        st.markdown("#### 👁️ Display Options")
        show_icons = st.checkbox("Show crop emojis", value=st.session_state.settings.get('show_crop_icons', True))
        st.session_state.settings['show_crop_icons'] = show_icons
        show_progress = st.checkbox("Show progress bars", value=st.session_state.settings.get('show_progress_bars', True))
        st.session_state.settings['show_progress_bars'] = show_progress
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reset Button
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.settings = {
            'dark_mode': False,
            'crop_recommendations_count': 5,
            'suitability_threshold': 50,
            'export_format': 'Both CSV & PDF',
            'default_input_method': 'Manual Entry',
            'show_crop_icons': True,
            'show_progress_bars': True,
        }
        st.session_state.dark_mode = False
        st.success("Settings reset to defaults!", icon="🔄")
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

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