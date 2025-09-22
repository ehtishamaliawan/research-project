"""
Unit tests for the research project package.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_project import DataLoader, DataPreprocessor, DataAnalyzer, Visualizer


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.loader = DataLoader(self.test_dir)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    
    def test_save_and_load_csv(self):
        """Test saving and loading CSV files."""
        filename = 'test_data.csv'
        
        # Save data
        self.loader.save_csv(self.sample_data, filename)
        
        # Load data
        loaded_data = self.loader.load_csv(filename)
        
        # Check if data matches
        pd.testing.assert_frame_equal(self.sample_data, loaded_data)
    
    def test_get_data_info(self):
        """Test getting data file information."""
        filename = 'test_info.csv'
        self.loader.save_csv(self.sample_data, filename)
        
        info = self.loader.get_data_info()
        self.assertIn(filename, info)
        self.assertEqual(info[filename]['type'], '.csv')


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data with missing values
        self.sample_data = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, np.nan, 5.5],
            'categorical': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_clean_data(self):
        """Test basic data cleaning."""
        # Add duplicate rows
        dirty_data = pd.concat([self.sample_data, self.sample_data.iloc[:2]])
        
        cleaned = self.preprocessor.clean_data(dirty_data)
        
        # Should have original number of rows after deduplication
        self.assertEqual(len(cleaned), len(self.sample_data))
    
    def test_handle_missing_values(self):
        """Test missing value imputation."""
        processed = self.preprocessor.handle_missing_values(
            self.sample_data, strategy='mean'
        )
        
        # Should have no missing values in numeric columns
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        self.assertFalse(processed[numeric_cols].isnull().any().any())
    
    def test_encode_categorical(self):
        """Test categorical encoding."""
        encoded = self.preprocessor.encode_categorical(
            self.sample_data, method='label'
        )
        
        # Categorical column should be numeric after encoding
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded['categorical']))
    
    def test_scale_features(self):
        """Test feature scaling."""
        scaled = self.preprocessor.scale_features(
            self.sample_data, method='standard'
        )
        
        # Numeric columns should have mean close to 0 after standard scaling
        numeric_cols = self.sample_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not scaled[col].isnull().all():
                self.assertAlmostEqual(scaled[col].dropna().mean(), 0, places=10)


class TestDataAnalyzer(unittest.TestCase):
    """Test cases for DataAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = DataAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'category': np.random.choice(['A', 'B'], 100),
            'target': np.random.normal(10, 3, 100)
        })
    
    def test_descriptive_statistics(self):
        """Test descriptive statistics generation."""
        stats = self.analyzer.descriptive_statistics(self.sample_data)
        
        self.assertIn('basic_info', stats)
        self.assertIn('missing_values', stats)
        self.assertIn('numeric_summary', stats)
        self.assertEqual(stats['basic_info']['shape'], self.sample_data.shape)
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        corr_matrix = self.analyzer.correlation_analysis(self.sample_data)
        
        # Correlation matrix should be square
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
        
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        outliers = self.analyzer.outlier_detection(self.sample_data)
        
        # Should return outlier info for numeric columns
        numeric_cols = self.sample_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertIn(col, outliers)
            self.assertIn('count', outliers[col])
            self.assertIn('percentage', outliers[col])


class TestVisualizer(unittest.TestCase):
    """Test cases for Visualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = Visualizer()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.normal(10, 3, 100)
        })
    
    def test_plot_distribution(self):
        """Test distribution plotting."""
        fig = self.visualizer.plot_distribution(self.sample_data, 'feature1')
        
        # Should return a matplotlib figure
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_plot_correlation_matrix(self):
        """Test correlation matrix plotting."""
        fig = self.visualizer.plot_correlation_matrix(self.sample_data)
        
        # Should return a matplotlib figure
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_create_dashboard(self):
        """Test dashboard creation."""
        dashboard = self.visualizer.create_dashboard(self.sample_data, 'target')
        
        # Should return a dictionary of plots
        self.assertIsInstance(dashboard, dict)
        self.assertGreater(len(dashboard), 0)


if __name__ == '__main__':
    unittest.main()