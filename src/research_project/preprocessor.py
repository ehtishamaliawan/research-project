"""
Data preprocessing utilities for research project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing utilities for research data.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        logger.info("DataPreprocessor initialized")
    
    def clean_data(self, df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates if requested
        if drop_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Remove rows that are entirely NaN
        df_clean = df_clean.dropna(how='all')
        
        # Remove columns that are entirely NaN
        df_clean = df_clean.dropna(axis=1, how='all')
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean', 
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'constant')
            columns: Specific columns to impute (if None, applies to all numeric columns)
            
        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        if columns is None:
            columns = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df_imputed.columns:
                if strategy in ['mean', 'median']:
                    imputer = SimpleImputer(strategy=strategy)
                elif strategy == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                
                df_imputed[[col]] = imputer.fit_transform(df_imputed[[col]])
                self.imputers[col] = imputer
                
        logger.info(f"Missing values handled for columns: {columns}")
        return df_imputed
    
    def encode_categorical(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                          method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (if None, applies to all object columns)
            method: Encoding method ('label', 'onehot')
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df_encoded.columns:
                if method == 'label':
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = encoder
                elif method == 'onehot':
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
        
        logger.info(f"Categorical encoding completed for columns: {columns}")
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (if None, applies to all numeric columns)
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df_scaled.columns:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    logger.warning(f"Unknown scaling method: {method}. Using standard scaling.")
                    scaler = StandardScaler()
                
                df_scaled[[col]] = scaler.fit_transform(df_scaled[[col]])
                self.scalers[col] = scaler
        
        logger.info(f"Feature scaling completed for columns: {columns}")
        return df_scaled
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_features = df.copy()
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        
        # Create polynomial features for numeric columns
        for col in numeric_columns:
            if col in df_features.columns:
                df_features[f'{col}_squared'] = df_features[col] ** 2
                df_features[f'{col}_log'] = np.log1p(np.abs(df_features[col]))
        
        # Create interaction features between numeric columns
        if len(numeric_columns) > 1:
            for i, col1 in enumerate(numeric_columns[:3]):  # Limit to avoid explosion
                for col2 in numeric_columns[i+1:4]:
                    if col1 != col2:
                        df_features[f'{col1}_{col2}_interaction'] = df_features[col1] * df_features[col2]
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing operations performed.
        
        Returns:
            Dictionary containing preprocessing summary
        """
        return {
            'scalers_fitted': list(self.scalers.keys()),
            'encoders_fitted': list(self.encoders.keys()),
            'imputers_fitted': list(self.imputers.keys())
        }