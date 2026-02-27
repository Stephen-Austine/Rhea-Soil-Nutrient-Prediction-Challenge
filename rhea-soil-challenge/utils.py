"""
Utility functions for Rhea Soil Nutrient Prediction Challenge
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json


def validate_submission(df, required_nutrients):
    """
    Validate submission file format per competition rules
    
    Args:
        df: Submission dataframe
        required_nutrients: List of nutrient names
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    required_cols = ['ID'] + [f'Target_{n}' for n in required_nutrients]
    
    # Check required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # Check for NaN values
    if df[required_cols].isnull().any().any():
        errors.append("Submission contains NaN values")
    
    # Check for infinite values
    if np.isinf(df[required_cols].values).any():
        errors.append("Submission contains infinite values")
    
    # Check ID uniqueness
    if df['ID'].duplicated().any():
        errors.append("Duplicate IDs found")
    
    # Check value ranges (should be non-negative)
    nutrient_cols = [f'Target_{n}' for n in required_nutrients]
    if (df[nutrient_cols] < 0).any().any():
        errors.append("Negative values found (nutrients should be >= 0)")
    
    return len(errors) == 0, errors


def apply_zero_mask(submission_df, mask_df, nutrients):
    """
    Apply TargetPred_To_Keep constraints: set masked predictions to exactly 0
    
    Args:
        submission_df: Submission dataframe with predictions
        mask_df: TargetPred_To_Keep dataframe
        nutrients: List of nutrient names
    
    Returns:
        DataFrame with zero-mask applied
    """
    submission = submission_df.copy()
    
    # Create a dictionary for fast lookup
    mask_dict = {}
    for _, row in mask_df.iterrows():
        sample_id = row['ID']
        for i, nutrient in enumerate(nutrients):
            # Columns in mask are: ID, then 13 nutrient columns (0 or 1)
            if i + 1 < len(row):
                if row.iloc[i + 1] == 0:
                    mask_dict[(sample_id, nutrient)] = True
    
    # Apply mask
    for nutrient in nutrients:
        target_col = f'Target_{nutrient}'
        for idx, row in submission.iterrows():
            if (row['ID'], nutrient) in mask_dict:
                submission.loc[idx, target_col] = 0.0
    
    return submission


def load_zero_mask(mask_path):
    """
    Load TargetPred_To_Keep.csv and convert to dictionary
    
    Args:
        mask_path: Path to TargetPred_To_Keep.csv
    
    Returns:
        Dictionary: {(ID, nutrient): should_be_zero}
    """
    mask_df = pd.read_csv(mask_path)
    nutrients = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']
    
    zero_mask_dict = {}
    for _, row in mask_df.iterrows():
        sample_id = row['ID']
        for i, nutrient in enumerate(nutrients):
            if i + 1 < len(row):
                if row.iloc[i + 1] == 0:
                    zero_mask_dict[(sample_id, nutrient)] = True
    
    return zero_mask_dict


def estimate_rmse(y_true, y_pred, sample_weights=None):
    """
    Calculate RMSE with optional weighting
    
    Args:
        y_true: True values
        y_pred: Predicted values
        sample_weights: Optional sample weights
    
    Returns:
        float: RMSE value
    """
    if sample_weights is None:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    else:
        return np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weights))


def create_feature_report(df, nutrient_cols):
    """
    Generate EDA report for feature engineering insights
    
    Args:
        df: DataFrame with soil data
        nutrient_cols: List of nutrient column names
    
    Returns:
        dict: Feature statistics and insights
    """
    report = {}
    
    # Basic stats
    report['shape'] = df.shape
    report['missing'] = df[nutrient_cols].isnull().sum().to_dict()
    
    # Distribution stats per nutrient
    for col in nutrient_cols:
        if col in df.columns and df[col].notna().any():
            report[f'{col}_stats'] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }
    
    # Correlation matrix (sample for large datasets)
    if len(df) > 10000:
        sample = df[nutrient_cols].sample(10000, random_state=42)
    else:
        sample = df[nutrient_cols]
    
    # Convert to serializable format
    corr_dict = {}
    corr_matrix = sample.corr()
    for col in corr_matrix.columns:
        corr_dict[col] = corr_matrix[col].to_dict()
    
    report['correlations'] = corr_dict
    
    return report


def save_metadata(metadata, path='models/metadata.json'):
    """Save model metadata to JSON"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, default=str, indent=2)


def load_metadata(path='models/metadata.json'):
    """Load model metadata from JSON"""
    with open(path, 'r') as f:
        return json.load(f)