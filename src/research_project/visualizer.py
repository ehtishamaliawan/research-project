"""
Data visualization utilities for research project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization utilities for research data analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the Visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        logger.info("Visualizer initialized")
    
    def plot_distribution(self, df: pd.DataFrame, column: str, 
                         bins: int = 30, show_stats: bool = True) -> plt.Figure:
        """
        Plot distribution of a single variable.
        
        Args:
            df: Input DataFrame
            column: Column name to plot
            bins: Number of bins for histogram
            show_stats: Whether to show statistical information
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'Distribution Analysis: {column}', fontsize=16)
        
        data = df[column].dropna()
        
        # Histogram
        axes[0, 0].hist(data, bins=bins, alpha=0.7, color=self.color_palette[0])
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(data)
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(column)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        
        # Statistics text
        if show_stats:
            stats_text = f"""
            Mean: {data.mean():.3f}
            Median: {data.median():.3f}
            Std: {data.std():.3f}
            Skewness: {stats.skew(data):.3f}
            Kurtosis: {stats.kurtosis(data):.3f}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        logger.info(f"Distribution plot created for {column}")
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson',
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: Input DataFrame
            method: Correlation method
            figsize: Figure size (if None, uses default)
            
        Returns:
            matplotlib Figure object
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr(method=method)
        
        if figsize is None:
            figsize = self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f')
        
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=16)
        plt.tight_layout()
        
        logger.info(f"Correlation matrix plotted using {method} method")
        return fig
    
    def plot_scatter_matrix(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                           target: Optional[str] = None) -> plt.Figure:
        """
        Create scatter plot matrix for numeric variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to include (if None, uses all numeric)
            target: Target variable for coloring points
            
        Returns:
            matplotlib Figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(columns)
        if n_cols > 6:
            columns = columns[:6]  # Limit to prevent overcrowding
            logger.warning("Limited to first 6 columns to prevent overcrowding")
        
        fig, axes = plt.subplots(n_cols, n_cols, figsize=(4*n_cols, 4*n_cols))
        fig.suptitle('Scatter Plot Matrix', fontsize=16)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                ax = axes[i, j] if n_cols > 1 else axes
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(df[col1].dropna(), bins=20, alpha=0.7, color=self.color_palette[0])
                    ax.set_title(col1)
                else:
                    # Off-diagonal: scatter plot
                    if target and target in df.columns:
                        for idx, target_val in enumerate(df[target].unique()[:5]):
                            mask = df[target] == target_val
                            ax.scatter(df.loc[mask, col2], df.loc[mask, col1], 
                                     alpha=0.6, label=str(target_val),
                                     color=self.color_palette[idx])
                        if i == 0 and j == n_cols - 1:
                            ax.legend()
                    else:
                        ax.scatter(df[col2], df[col1], alpha=0.6, color=self.color_palette[0])
                
                if i == n_cols - 1:
                    ax.set_xlabel(col2)
                if j == 0:
                    ax.set_ylabel(col1)
        
        plt.tight_layout()
        logger.info("Scatter plot matrix created")
        return fig
    
    def plot_categorical_analysis(self, df: pd.DataFrame, column: str,
                                target: Optional[str] = None) -> plt.Figure:
        """
        Analyze categorical variable with various plots.
        
        Args:
            df: Input DataFrame
            column: Categorical column to analyze
            target: Target variable for cross-analysis
            
        Returns:
            matplotlib Figure object
        """
        if target:
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        fig.suptitle(f'Categorical Analysis: {column}', fontsize=16)
        
        # Value counts
        value_counts = df[column].value_counts()
        
        if target:
            # Bar plot
            axes[0, 0].bar(range(len(value_counts)), value_counts.values, color=self.color_palette[0])
            axes[0, 0].set_xticks(range(len(value_counts)))
            axes[0, 0].set_xticklabels(value_counts.index, rotation=45)
            axes[0, 0].set_title('Value Counts')
            axes[0, 0].set_ylabel('Count')
            
            # Pie chart
            axes[0, 1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Distribution')
            
            # Cross-tabulation with target
            if target in df.columns:
                crosstab = pd.crosstab(df[column], df[target])
                crosstab.plot(kind='bar', stacked=True, ax=axes[1, 0])
                axes[1, 0].set_title(f'{column} vs {target}')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].legend(title=target)
                
                # Normalized cross-tabulation
                crosstab_norm = pd.crosstab(df[column], df[target], normalize='index')
                crosstab_norm.plot(kind='bar', stacked=True, ax=axes[1, 1])
                axes[1, 1].set_title(f'{column} vs {target} (Normalized)')
                axes[1, 1].set_ylabel('Proportion')
                axes[1, 1].legend(title=target)
        else:
            # Bar plot
            axes[0].bar(range(len(value_counts)), value_counts.values, color=self.color_palette[0])
            axes[0].set_xticks(range(len(value_counts)))
            axes[0].set_xticklabels(value_counts.index, rotation=45)
            axes[0].set_title('Value Counts')
            axes[0].set_ylabel('Count')
            
            # Pie chart
            axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            axes[1].set_title('Distribution')
        
        plt.tight_layout()
        logger.info(f"Categorical analysis plot created for {column}")
        return fig
    
    def plot_time_series(self, df: pd.DataFrame, date_column: str, 
                        value_columns: List[str], resample: Optional[str] = None) -> go.Figure:
        """
        Create interactive time series plot.
        
        Args:
            df: Input DataFrame
            date_column: Date column name
            value_columns: List of value columns to plot
            resample: Resampling frequency ('D', 'W', 'M', etc.)
            
        Returns:
            Plotly Figure object
        """
        df_ts = df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        df_ts = df_ts.set_index(date_column).sort_index()
        
        if resample:
            df_ts = df_ts.resample(resample).mean()
        
        fig = go.Figure()
        
        for i, col in enumerate(value_columns):
            if col in df_ts.columns:
                fig.add_trace(go.Scatter(
                    x=df_ts.index,
                    y=df_ts[col],
                    mode='lines',
                    name=col,
                    line=dict(color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)])
                ))
        
        fig.update_layout(
            title='Time Series Analysis',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True
        )
        
        logger.info("Interactive time series plot created")
        return fig
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                               top_n: int = 15) -> plt.Figure:
        """
        Plot feature importance from model results.
        
        Args:
            importance_dict: Dictionary with feature names and importance scores
            top_n: Number of top features to display
            
        Returns:
            matplotlib Figure object
        """
        if isinstance(importance_dict, dict) and 'feature' in importance_dict:
            # Handle pandas DataFrame dict format
            features = importance_dict['feature']
            importances = importance_dict['importance']
            importance_df = pd.DataFrame({
                'feature': list(features.values()),
                'importance': list(importances.values())
            })
        else:
            # Handle simple dict format
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} for k, v in importance_dict.items()
            ])
        
        importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                      color=self.color_palette[0])
        
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, importance_df['importance']):
            ax.text(value, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        logger.info("Feature importance plot created")
        return fig
    
    def create_dashboard(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive dashboard of visualizations.
        
        Args:
            df: Input DataFrame
            target_col: Target column for analysis
            
        Returns:
            Dictionary containing all generated plots
        """
        dashboard = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            dashboard['correlation_matrix'] = self.plot_correlation_matrix(df)
        
        # Distribution plots for numeric columns (first 4)
        dashboard['distributions'] = {}
        for col in numeric_cols[:4]:
            dashboard['distributions'][col] = self.plot_distribution(df, col)
        
        # Categorical analysis (first 2)
        dashboard['categorical'] = {}
        for col in categorical_cols[:2]:
            dashboard['categorical'][col] = self.plot_categorical_analysis(df, col, target_col)
        
        # Scatter matrix for top numeric columns
        if len(numeric_cols) > 1:
            top_numeric = numeric_cols[:4]
            dashboard['scatter_matrix'] = self.plot_scatter_matrix(df, top_numeric, target_col)
        
        logger.info("Comprehensive dashboard created")
        return dashboard