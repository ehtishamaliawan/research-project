"""
Data analysis utilities for research project.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, regression_metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Comprehensive data analysis utilities for research.
    """
    
    def __init__(self):
        """Initialize the DataAnalyzer."""
        self.models = {}
        self.analysis_results = {}
        logger.info("DataAnalyzer initialized")
    
    def descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive descriptive statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing descriptive statistics
        """
        results = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }
        
        # Categorical variable summaries
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            results['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head().to_dict()
            }
        
        logger.info("Descriptive statistics generated")
        self.analysis_results['descriptive_stats'] = results
        return results
    
    def correlation_analysis(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Perform correlation analysis on numeric variables.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation analysis")
            return pd.DataFrame()
        
        correlation_matrix = numeric_df.corr(method=method)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        self.analysis_results['correlation'] = {
            'matrix': correlation_matrix,
            'high_correlations': high_corr_pairs
        }
        
        logger.info(f"Correlation analysis completed using {method} method")
        return correlation_matrix
    
    def statistical_tests(self, df: pd.DataFrame, target_col: str, 
                         test_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform statistical tests between variables and target.
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
            test_cols: Columns to test against target (if None, uses all numeric)
            
        Returns:
            Dictionary containing test results
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        if test_cols is None:
            test_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in test_cols:
                test_cols.remove(target_col)
        
        results = {}
        target_values = df[target_col].dropna()
        
        for col in test_cols:
            if col in df.columns and col != target_col:
                col_values = df[col].dropna()
                
                # Correlation test
                if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                    corr_coef, corr_p_value = stats.pearsonr(
                        df[[col, target_col]].dropna()[col], 
                        df[[col, target_col]].dropna()[target_col]
                    )
                    
                    # T-test for means
                    if len(np.unique(target_values)) == 2:  # Binary target
                        group1 = df[df[target_col] == np.unique(target_values)[0]][col].dropna()
                        group2 = df[df[target_col] == np.unique(target_values)[1]][col].dropna()
                        if len(group1) > 0 and len(group2) > 0:
                            t_stat, t_p_value = stats.ttest_ind(group1, group2)
                        else:
                            t_stat, t_p_value = np.nan, np.nan
                    else:
                        t_stat, t_p_value = np.nan, np.nan
                    
                    results[col] = {
                        'correlation': corr_coef,
                        'correlation_p_value': corr_p_value,
                        't_statistic': t_stat,
                        't_test_p_value': t_p_value
                    }
        
        logger.info(f"Statistical tests completed for {len(results)} variables")
        self.analysis_results['statistical_tests'] = results
        return results
    
    def quick_model_evaluation(self, df: pd.DataFrame, target_col: str, 
                              task_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform quick model evaluation to assess predictive potential.
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
            task_type: 'classification', 'regression', or 'auto' to detect
            
        Returns:
            Dictionary containing model evaluation results
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Prepare data
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df[target_col].dropna()
        
        # Align X and y indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if X.empty:
            logger.error("No numeric features available for modeling")
            return {}
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Determine task type
        if task_type == 'auto':
            if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
                task_type = 'regression'
            else:
                task_type = 'classification'
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {'task_type': task_type}
        
        try:
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                results['accuracy'] = (y_pred == y_test).mean()
                results['classification_report'] = classification_report(
                    y_test, y_pred, output_dict=True
                )
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['feature_importance'] = feature_importance.head(10).to_dict()
                
            elif task_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                results['mse'] = np.mean((y_pred - y_test) ** 2)
                results['rmse'] = np.sqrt(results['mse'])
                results['r2'] = 1 - (np.sum((y_test - y_pred) ** 2) / 
                                   np.sum((y_test - np.mean(y_test)) ** 2))
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['feature_importance'] = feature_importance.head(10).to_dict()
            
            self.models[f'{target_col}_{task_type}'] = model
            logger.info(f"Quick model evaluation completed for {task_type} task")
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            results['error'] = str(e)
        
        self.analysis_results['model_evaluation'] = results
        return results
    
    def outlier_detection(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr', 'zscore')
            
        Returns:
            Dictionary containing outlier information
        """
        numeric_df = df.select_dtypes(include=[np.number])
        outlier_info = {}
        
        for col in numeric_df.columns:
            column_data = numeric_df[col].dropna()
            
            if method == 'iqr':
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(column_data))
                outliers = column_data[z_scores > 3]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(column_data) * 100,
                'values': outliers.tolist()[:10]  # Show first 10 outliers
            }
        
        logger.info(f"Outlier detection completed using {method} method")
        self.analysis_results['outliers'] = outlier_info
        return outlier_info
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of all analyses performed.
        
        Returns:
            Dictionary containing analysis summary
        """
        return {
            'analyses_performed': list(self.analysis_results.keys()),
            'models_trained': list(self.models.keys()),
            'results': self.analysis_results
        }