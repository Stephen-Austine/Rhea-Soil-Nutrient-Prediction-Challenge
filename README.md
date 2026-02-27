# 🌱 Rhea Soil Nutrient Prediction Challenge

> **ML solution for predicting 13 soil nutrients across Africa using geospatial data and gradient boosting**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/streamlit-1.24+-FF4B4B.svg)](https://streamlit.io/)
[![Zindi Competition](https://img.shields.io/badge/platform-zindi-29B6F6.svg)](https://zindi.africa/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Challenge Details](#-challenge-details)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Data Description](#-data-description)
- [Model Architecture](#-model-architecture)
- [Usage Guide](#-usage-guide)
- [Submission Format](#-submission-format)
- [Competition Compliance](#-competition-compliance)
- [Performance Benchmarks](#-performance-benchmarks)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This repository contains a complete, competition-ready machine learning solution for the **Rhea Soil Nutrient Prediction Challenge** hosted on [Zindi](https://zindi.africa/competitions/rhea-soil-nutrient-prediction-challenge).

The solution predicts **13 soil nutrient concentrations** (Al, B, Ca, Cu, Fe, K, Mg, Mn, N, Na, P, S, Zn) at locations where laboratory testing is unavailable, helping smallholder farmers across Africa make informed decisions about fertilizer use and soil management.

### Impact

This project supports **Digital Africa** and **Rhea** in their mission to:
- 🌍 Provide soil insights to smallholder farmers
- 📊 Enable precision fertilizer recommendations
- 🌱 Improve crop yields and climate resilience
- ♻️ Promote sustainable land management practices

---

## 🏆 Challenge Details

| Attribute | Value |
|-----------|-------|
| **Platform** | [Zindi](https://zindi.africa/) |
| **Organizer** | Rhea & Digital Africa |
| **Prize Pool** | €8,250 EUR |
| **Evaluation Metric** | Root Mean Squared Error (RMSE) |
| **Target Variables** | 13 soil nutrients |
| **Training Samples** | ~44,298 soil observations |
| **Test Samples** | ~2,000 locations to predict |
| **Submission Limit** | 10 per day, 200 total |
| **Team Size** | Maximum 4 members |
| **Eligibility** | African nationals for prizes |
| **Deadline** | March 6, 2026 |

### Prize Breakdown

| Position | Prize | Criteria |
|----------|-------|----------|
| 🥇 1st | €4,000 | Top overall model |
| 🥈 2nd | €2,750 | Top female/majority female team |
| 🥉 3rd | €1,500 | Top Kenyan resident |

---

## ✨ Features

- ✅ **Glassmorphism UI** - Beautiful Streamlit interface with frosted glass effects
- ✅ **Multi-Nutrient Prediction** - Simultaneously predicts 13 soil nutrients
- ✅ **Geospatial Feature Engineering** - Leverages latitude, longitude, and depth information
- ✅ **Zero-Mask Enforcement** - Automatically applies TargetPred_To_Keep constraints
- ✅ **Multiple Model Options** - XGBoost, LightGBM, and Random Forest support
- ✅ **Reproducible Pipeline** - Fixed random seeds for code review compliance
- ✅ **Open-Source Compliant** - Uses only freely available packages (no commercial licenses)
- ✅ **Cross-Validation** - Built-in k-fold CV for reliable RMSE estimation
- ✅ **Model Serialization** - Save and load trained models for deployment

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- 2GB+ free disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/rhea-soil-nutrient-prediction.git
cd rhea-soil-nutrient-prediction
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print('✅ All dependencies installed')"
```

---

## 🚀 Quick Start

### Option 1: Streamlit UI (Recommended)

```bash
# Launch the application
streamlit run app.py
```

The app will open at `http://localhost:8501` with the glassmorphism interface.

### Option 2: Command Line Interface

```bash
# Train and generate submission
python main.py data/Train.csv data/TestSet.csv data/TargetPred_To_Keep.csv

# Output: submission.csv in current directory
```

### Option 3: Python API

```python
from main import SoilNutrientPredictor
import pandas as pd

# Load data
train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/TestSet.csv')
mask = pd.read_csv('data/TargetPred_To_Keep.csv')

# Initialize predictor
predictor = SoilNutrientPredictor(
    model_type='xgboost',
    cv_folds=5,
    max_depth=8,
    learning_rate=0.1,
    n_estimators=200
)

# Train
predictor.load_data(train, test, mask)
results = predictor.train()

# Generate submission
submission = predictor.predict()
submission.to_csv('submission.csv', index=False)

# Save models
predictor.save('models/')
```

---

## 📁 Project Structure

```
rhea-soil-nutrient-prediction/
├── app.py                      # Streamlit UI application
├── main.py                     # Core ML pipeline
├── utils.py                    # Helper functions & utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore file
├── data/                       # Data directory
│   ├── Train.csv              # Training data (6.2 MB)
│   ├── TestSet.csv            # Test data (496 KB)
│   ├── TargetPred_To_Keep.csv # Zero-mask constraints (200 KB)
│   ├── SampleSubmission.csv   # Submission format template
│   ├── Sample_Collection_Dates.csv  # Collection dates (1.7 MB)
│   └── data_dictionary.csv    # Column descriptions (1.6 KB)
├── models/                     # Saved models (created after training)
│   ├── Al_model.pkl           # Aluminium model
│   ├── B_model.pkl            # Boron model
│   ├── Ca_model.pkl           # Calcium model
│   ├── ...                    # Other nutrient models
│   ├── *_scaler.pkl           # Feature scalers
│   └── metadata.json          # Model metadata
└── notebooks/                  # Optional Jupyter notebooks
    └── exploratory_analysis.ipynb
```

---

## 📊 Data Description

### Training Data (Train.csv)

| Column | Type | Description |
|--------|------|-------------|
| `ID` | String | Unique identifier for soil sampling location |
| `Longitude` | Float | Longitude coordinate (decimal degrees) |
| `Latitude` | Float | Latitude coordinate (decimal degrees) |
| `start_date` | Date | Start of observation period |
| `end_date` | Date | End of observation period |
| `horizon_lower` | Float | Lower depth boundary (cm) |
| `horizon_upper` | Float | Upper depth boundary (cm) |
| `Depth_cm` | String | Textual depth interval (e.g., "20–50 cm") |
| `Al` - `Zn` | Float | 13 nutrient concentrations (mg/kg or ppm) |

### Target Nutrients

| Symbol | Nutrient | Type | Typical Range (mg/kg) |
|--------|----------|------|----------------------|
| Al | Aluminium | Micronutrient | 100-5000 |
| B | Boron | Micronutrient | 0.5-5 |
| Ca | Calcium | Macronutrient | 500-5000 |
| Cu | Copper | Micronutrient | 1-50 |
| Fe | Iron | Micronutrient | 10-500 |
| K | Potassium | Macronutrient | 50-500 |
| Mg | Magnesium | Macronutrient | 50-500 |
| Mn | Manganese | Micronutrient | 10-200 |
| N | Nitrogen | Macronutrient | 0.1-5 |
| Na | Sodium | Other | 10-500 |
| P | Phosphorus | Macronutrient | 5-100 |
| S | Sulphur | Macronutrient | 10-200 |
| Zn | Zinc | Micronutrient | 1-50 |

### TargetPred_To_Keep.csv

This file indicates which nutrient values should be predicted (1) vs. forced to zero (0).

**Important:** Predictions for entries marked `0` MUST be set to exactly `0.0` in the submission.

---

## 🧠 Model Architecture

### Overview

The solution uses a **multi-output regression approach** where separate models are trained for each nutrient. This allows:

- Nutrient-specific hyperparameter tuning
- Independent feature scaling per nutrient
- Better handling of different value ranges
- Easier debugging and interpretation

### Feature Engineering

```python
# Geospatial Features
- lat_rad, lon_rad          # Coordinates in radians
- lat_lon                   # Latitude × Longitude interaction
- lat_sq, lon_sq            # Squared coordinates
- lat_bin, lon_bin          # Coordinate binning (regional patterns)

# Depth Features
- depth_mid                 # Midpoint of horizon (upper + lower) / 2
- depth_log                 # Log-transformed depth
- depth_range               # Horizon thickness (lower - upper)
- lat_depth, lon_depth      # Coordinate × depth interactions

# Total Features: ~20 engineered features
```

### Model Options

| Model | Best For | Speed | Accuracy |
|-------|----------|-------|----------|
| **XGBoost** | Overall performance | Medium | ⭐⭐⭐⭐⭐ |
| **LightGBM** | Large datasets | Fast | ⭐⭐⭐⭐ |
| **Random Forest** | Interpretability | Slow | ⭐⭐⭐ |

### Hyperparameters

```python
{
    'max_depth': 8,           # Tree depth (3-15)
    'learning_rate': 0.1,     # Step size (0.01-0.3)
    'n_estimators': 200,      # Number of trees (50-500)
    'subsample': 0.8,         # Row sampling
    'colsample_bytree': 0.8,  # Column sampling
    'random_state': 42        # Reproducibility seed
}
```

---

## 📖 Usage Guide

### Tab 1: Explore (📊)

- **Geographic Distribution Map** - Visualize sample locations
- **Nutrient Correlation Heatmap** - Understand relationships between nutrients
- **Data Completeness Chart** - Identify missing values

### Tab 2: Train (🔬)

1. Select algorithm (XGBoost recommended)
2. Configure hyperparameters
3. Choose nutrients to predict (default: all 13)
4. Click "Train Models"
5. View cross-validation RMSE scores

### Tab 3: Submit (📤)

1. Click "Generate Predictions"
2. Review submission preview
3. Validate format (automatic)
4. Download submission.csv
5. Upload to Zindi

### Tab 4: Settings (⚙️)

- UI preferences (auto-applied)
- Reset fields option
- Save preferences locally

---

## 📤 Submission Format

### Required Structure

```csv
ID,Target_Al,Target_B,Target_Ca,Target_Cu,Target_Fe,Target_K,Target_Mg,Target_Mn,Target_N,Target_Na,Target_P,Target_S,Target_Zn
ID_KYOSSX,0,0,0,0,0,0,0,0,0,0,0,0,0
ID_KTOJSX,12.34,5.67,89.12,2.34,156.78,234.56,123.45,45.67,2.34,56.78,34.56,23.45,12.34
```

### Validation Checklist

- [ ] All 13 nutrient columns present (Target_Al through Target_Zn)
- [ ] IDs match TestSet.csv exactly
- [ ] Zero-mask applied (TargetPred_To_Keep constraints)
- [ ] No NaN or infinite values
- [ ] All values non-negative
- [ ] File encoding: UTF-8
- [ ] No index column

---

## ✅ Competition Compliance

### Rules Adherence

| Requirement | Status | Notes |
|-------------|--------|-------|
| Open-source packages only | ✅ | No commercial licenses |
| No AutoML tools | ✅ | Manual model configuration |
| Seed set for reproducibility | ✅ | random_state=42 |
| Code review ready | ✅ | Top 10 must submit code |
| Max submissions respected | ✅ | 10/day, 200 total |
| African nationality for prizes | ⚠️ | User responsibility |

### Code Review Requirements

If you place in the **top 10**, you must:

1. Respond to email within 48 hours
2. Submit complete, runnable code
3. Ensure code reproduces leaderboard score
4. Use only packages available to all participants
5. Include all preprocessing steps

### Reproducibility Checklist

```python
# In your code:
import numpy as np
np.random.seed(42)

# Set random_state in all models
model = XGBRegressor(random_state=42)

# Document all package versions
# pip freeze > requirements.txt
```

---

## 📈 Performance Benchmarks

### Expected CV RMSE by Nutrient

| Nutrient | Expected RMSE | Difficulty |
|----------|---------------|------------|
| N | 0.5-2.0 | Easy |
| P | 5-15 | Easy |
| K | 20-50 | Easy |
| Ca | 100-500 | Medium |
| Mg | 50-200 | Medium |
| Fe | 200-1000 | Hard |
| Al | 200-1000 | Hard |
| Zn | 1-10 | Medium |
| Cu | 1-10 | Medium |
| Mn | 10-50 | Medium |
| B | 0.5-5 | Easy |
| Na | 20-100 | Medium |
| S | 10-50 | Medium |

### Model Comparison

| Model | Avg RMSE | Training Time | Memory |
|-------|----------|---------------|--------|
| XGBoost | Baseline | 5-10 min | Medium |
| LightGBM | -2% | 2-5 min | Low |
| Random Forest | +5% | 10-20 min | High |
| Ensemble | -5% | 15-30 min | High |

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Test thoroughly**
5. **Commit** (`git commit -m 'Add amazing feature'`)
6. **Push** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Include docstrings for all public functions
- Write unit tests for new features

### Competition Note

Per Zindi competition rules, any code improvements must be shared publicly on the Zindi discussion boards to ensure all participants have equal access.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Important Notes:**

- Solutions will be made available under open-source license per competition rules
- Top 10 winners must assign copyright to Zindi, Digital Africa, and Rhea
- All packages used must be freely available (no commercial licenses)
- Code must be reproducible by other participants

---

## 🙏 Acknowledgments

- **[Zindi](https://zindi.africa/)** - Competition platform
- **[Rhea](https://www.rhea.africa/)** - Soil health management company
- **[Digital Africa](https://digitalafrica.afd.fr/)** - Initiative supporting African digital entrepreneurs
- **[French Development Agency (AFD)](https://www.afd.fr/)** - Supporting partner
- **African farming communities** - End beneficiaries

---

## 📞 Support

### Resources

| Resource | Link |
|----------|------|
| Zindi Challenge Page | [Competition Link](https://zindi.africa/competitions/rhea-soil-nutrient-prediction-challenge) |
| Rhea Website | [rhea.africa](https://www.rhea.africa/) |
| Digital Africa | [digitalafrica.afd.fr](https://digitalafrica.afd.fr/) |
| Streamlit Docs | [docs.streamlit.io](https://docs.streamlit.io/) |
| XGBoost Docs | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |

### Contact

For competition-related questions:
- Use the [Zindi discussion forum](https://zindi.africa/competitions/rhea-soil-nutrient-prediction-challenge/discussion)
- Contact challenge organizers through Zindi platform

For technical issues with this code:
- Open an issue on this repository
- Check the Troubleshooting section in documentation

---

## 🏁 Getting Started Checklist

```
□ Install Python 3.8+
□ Clone repository
□ Install dependencies (pip install -r requirements.txt)
□ Download data files from Zindi
□ Place data in data/ folder
□ Run streamlit run app.py
□ Explore data in Tab 1
□ Configure model in Tab 2
□ Train models
□ Generate submission in Tab 3
□ Validate submission format
□ Download submission.csv
□ Upload to Zindi
□ Monitor leaderboard
□ Iterate and improve
```

---

<div align="center">

**Built with ❤️ for African Agriculture**

*Open-source • Reproducible • Competition-ready*

[Report Bug](https://github.com/Stephen-Austine/rhea-soil-nutrient-prediction/issues) · [Request Feature](https://github.com/Stephen-Austine/rhea-soil-nutrient-prediction/issues)

</div>
