"""
Rhea Soil Nutrient Prediction Challenge
Main ML Module - SoilNutrientPredictor Class
Author: Steve Ochwada | Feb 2026
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


class SoilNutrientPredictor:
    """
    Multi-output regressor for soil nutrient prediction.
    Supports XGBoost, LightGBM, and Random Forest algorithms.
    """
    
    def __init__(self, model_type='xgboost', cv_folds=5, max_depth=8, 
                 learning_rate=0.1, n_estimators=200, random_state=42):
        """
        Initialize the predictor.
        
        Args:
            model_type: 'xgboost', 'lightgbm', or 'random_forest'
            cv_folds: Number of cross-validation folds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for boosting
            n_estimators: Number of estimators/trees
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.models = {}
        self.feature_cols = []
        self.nutrients = []
        self.training_results = {}
        self.train_df = None
        self.test_df = None
        self.mask_df = None
        
    def load_data(self, train_df, test_df, mask_df):
        """
        Load and store the datasets.
        
        Args:
            train_df: Training data
            test_df: Test data
            mask_df: Target prediction mask (which nutrients to predict)
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.mask_df = mask_df.copy()
        
    def create_features(self, df):
        """
        Create engineered features from raw data.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # According to metadata, only these features were used for training
        # Set feature columns to exactly what was used during training
        self.feature_cols = ['Latitude', 'Longitude', 'Lat_Lon_Product', 'Lat_Lon_Sum', 
                           'Distance_From_Equator', 'Northern_Hemisphere', 'Eastern_Hemisphere']
        
        # Create only the features that were actually used
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Geographic features
            df['Lat_Lon_Product'] = df['Latitude'] * df['Longitude']
            df['Lat_Lon_Sum'] = df['Latitude'] + df['Longitude']
            
            # Distance from equator
            df['Distance_From_Equator'] = np.abs(df['Latitude'])
            
            # Hemisphere indicators
            df['Northern_Hemisphere'] = (df['Latitude'] >= 0).astype(int)
            df['Eastern_Hemisphere'] = (df['Longitude'] >= 0).astype(int)
        
        return df
    
    def get_model(self):
        """Get a fresh model instance based on model_type."""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            # Fallback to Random Forest if unknown model type
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def train(self, nutrients):
        """
        Train models for specified nutrients with cross-validation.
        
        Args:
            nutrients: List of nutrient column names to train on
            
        Returns:
            dict: Training results for each nutrient
        """
        self.nutrients = nutrients
        
        # Prepare training data
        train_data = self.create_features(self.train_df)
        self.train_data = train_data  # Store for use in predict()
        X = train_data[self.feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Replace infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        results = {}
        
        print(f"\nTraining {len(nutrients)} nutrient models using {self.model_type}...")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Training samples: {len(X)}")
        
        for i, nutrient in enumerate(nutrients):
            print(f"\nTraining model for {nutrient} ({i+1}/{len(nutrients)})")
            
            y = self.train_df[nutrient].copy()
            
            # Remove samples with missing target
            valid_idx = ~y.isna()
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]
            
            print(f"  Valid samples: {len(X_valid)} (dropped {sum(~valid_idx)} NaN targets)")
            
            # Cross-validation
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            cv_scores = []
            train_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_valid)):
                X_train, X_val = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
                y_train, y_val = y_valid.iloc[train_idx], y_valid.iloc[val_idx]
                
                model = self.get_model()
                model.fit(X_train, y_train)
                
                # Predictions
                val_pred = model.predict(X_val)
                train_pred = model.predict(X_train)
                
                # RMSE
                cv_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                
                cv_scores.append(cv_rmse)
                train_scores.append(train_rmse)
            
            # Train final model on all data
            final_model = self.get_model()
            final_model.fit(X_valid, y_valid)
            
            self.models[nutrient] = final_model
            
            results[nutrient] = {
                'cv_rmse': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'train_rmse': np.mean(train_scores)
            }
            
            print(f"  CV RMSE: {results[nutrient]['cv_rmse']:.4f} (+/- {results[nutrient]['cv_std']:.4f})")
            print(f"  Train RMSE: {results[nutrient]['train_rmse']:.4f}")
        
        self.training_results = results
        return results
    
    def predict(self, input_data=None):
        """
        Generate predictions for the test set or input data.
        
        Args:
            input_data: Optional DataFrame or dict for custom prediction
            
        Returns:
            DataFrame: Predictions
        """
        if input_data is None:
            # Default to test set
            test_data = self.create_features(self.test_df)
            X_test = test_data[self.feature_cols].copy()
            
            # Handle missing values (use training median from features)
            if hasattr(self, 'train_data') and self.train_data is not None:
                train_medians = self.train_data[self.feature_cols].median() if self.feature_cols else pd.Series()
            else:
                train_data = self.create_features(self.train_df)
                train_medians = train_data[self.feature_cols].median() if self.feature_cols else pd.Series()
            
            X_test = X_test.fillna(train_medians)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.fillna(0)
            
            predictions = {}
            predictions['ID'] = self.test_df['ID'].values if 'ID' in self.test_df.columns else np.arange(len(self.test_df))
            
        else:
            # Predict on input data
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Create features
            test_data = self.create_features(input_data)
            X_test = test_data[self.feature_cols].copy()
            
            # Handle missing values (use training median from features)
            if hasattr(self, 'train_data') and self.train_data is not None:
                train_medians = self.train_data[self.feature_cols].median() if self.feature_cols else pd.Series()
            else:
                train_data = self.create_features(self.train_df)
                train_medians = train_data[self.feature_cols].median() if self.feature_cols else pd.Series()
            
            X_test = X_test.fillna(train_medians)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.fillna(0)
            
            predictions = {}
            if 'ID' in input_data.columns:
                predictions['ID'] = input_data['ID'].values
            else:
                predictions['ID'] = np.arange(len(input_data))
        
        # Generate predictions
        for nutrient in self.nutrients:
            if nutrient in self.models:
                model = self.models[nutrient]
                pred = model.predict(X_test)
                
                # Ensure non-negative predictions
                pred = np.maximum(pred, 0)
                
                predictions[f'Target_{nutrient}'] = pred
        
        submission = pd.DataFrame(predictions)
        
        # Apply zero mask only to test set predictions
        if input_data is None and self.mask_df is not None:
            submission = self.apply_zero_mask(submission)
        
        return submission
    
    def predict_single(self, input_data):
        """
        Predict nutrient levels for a single soil sample.
        
        Args:
            input_data: Dictionary or Series containing soil properties
            
        Returns:
            dict: Predicted nutrient levels
        """
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        predictions = self.predict(input_data)
        
        # Convert to single sample results
        if len(predictions) > 0:
            result = predictions.iloc[0].to_dict()
            # Remove ID if present
            if 'ID' in result:
                del result['ID']
            # Strip 'Target_' prefix from nutrient keys
            stripped_result = {}
            for key, value in result.items():
                if key.startswith('Target_'):
                    stripped_result[key[len('Target_'):]] = value
                else:
                    stripped_result[key] = value
            return stripped_result
        return {}
    
    def apply_zero_mask(self, submission):
        """
        Apply the TargetPred_To_Keep mask to set certain predictions to 0.
        
        Args:
            submission: DataFrame with predictions
            
        Returns:
            DataFrame with zero-mask applied
        """
        if self.mask_df is None:
            return submission
        
        submission = submission.copy()
        
        # Get mask columns (ID + 13 nutrient columns)
        mask_cols = self.mask_df.columns.tolist()
        
        # For each nutrient, check if it should be masked
        for nutrient in self.nutrients:
            target_col = f'Target_{nutrient}'
            if target_col not in submission.columns:
                continue
            
            # Find the corresponding column in mask_df
            # The mask_df has columns like: ID, Al, B, Ca, etc.
            if nutrient in self.mask_df.columns:
                # Get IDs where mask is 0
                masked_ids = self.mask_df[self.mask_df[nutrient] == 0]['ID'].values
                
                # Set those predictions to 0
                submission.loc[submission['ID'].isin(masked_ids), target_col] = 0.0
        
        return submission
    
    def save(self, path='models/'):
        """
        Save the trained models and metadata.
        
        Args:
            path: Directory to save models
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save models
        for nutrient, model in self.models.items():
            joblib.dump(model, f"{path}model_{nutrient}.joblib")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'cv_folds': self.cv_folds,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'feature_cols': self.feature_cols,
            'nutrients': self.nutrients,
            'training_results': self.training_results
        }
        
        with open(f"{path}metadata.json", 'w') as f:
            json.dump(metadata, f, default=str, indent=2)
        
        print(f"Models saved to {path}")
    
    def load(self, path='models/'):
        """
        Load trained models and metadata.
        
        Args:
            path: Directory containing saved models
        """
        # Load metadata
        with open(f"{path}metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.cv_folds = metadata['cv_folds']
        self.max_depth = metadata['max_depth']
        self.learning_rate = metadata['learning_rate']
        self.n_estimators = metadata['n_estimators']
        self.feature_cols = metadata['feature_cols']
        self.nutrients = metadata['nutrients']
        self.training_results = metadata.get('training_results', {})
        
        # Load models
        for nutrient in self.nutrients:
            self.models[nutrient] = joblib.load(f"{path}model_{nutrient}.joblib")
        
        # Create complete dummy data with all original features
        dummy_data = {
            'Latitude': [0.0], 'Longitude': [0.0], 'Elevation': [0.0],
            'pH': [7.0], 'OrganicMatter': [2.5], 'Temperature': [25.0],
            'Sand': [50], 'Clay': [20], 'Silt': [30], 'EC': [0.5],
            'SoilMoisture': [25.0], 'CEC': [15.0]
        }
        
        self.train_df = pd.DataFrame(dummy_data)
        self.train_data = self.create_features(self.train_df)
        
        print(f"Models loaded from {path}")


def main():
    """Main function for standalone execution."""
    import sys
    
    print("=" * 60)
    print("Rhea Soil Nutrient Prediction - Model Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_csv("Train.csv")
    test = pd.read_csv("TestSet.csv")
    mask = pd.read_csv("TargetPred_To_Keep.csv")
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Define nutrients
    nutrients = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']
    
    # Initialize predictor
    predictor = SoilNutrientPredictor(
        model_type='xgboost',
        cv_folds=5,
        max_depth=8,
        learning_rate=0.1,
        n_estimators=200
    )
    
    # Load data
    predictor.load_data(train, test, mask)
    
    # Train models
    print("\nTraining models...")
    results = predictor.train(nutrients)
    
    # Print results summary
    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    for nutrient, metrics in results.items():
        print(f"{nutrient}: CV RMSE = {metrics['cv_rmse']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    # Generate predictions
    print("\nGenerating predictions...")
    submission = predictor.predict()
    
    # Save submission
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved to submission.csv")
    print(f"Submission shape: {submission.shape}")
    
    # Save models
    predictor.save("models/")
    print("\nDone!")


if __name__ == "__main__":
    main()

