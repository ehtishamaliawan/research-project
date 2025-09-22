"""
Example Jupyter notebook demonstrating research project capabilities.
"""

# This is a markdown cell that would be converted in Jupyter
"""
# Research Project Demo

This notebook demonstrates the capabilities of the research-project package for data analysis and machine learning.

## Setup
"""

# Import required libraries
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project src to path
project_root = Path('.').resolve()
if str(project_root / 'src') not in sys.path:
    sys.path.insert(0, str(project_root / 'src'))

# Import our research project modules
from research_project import DataLoader, DataPreprocessor, DataAnalyzer, Visualizer

"""
## Create Sample Data

Let's create some sample data to demonstrate the functionality.
"""

# Create sample research data
np.random.seed(42)
n_samples = 1000

# Generate synthetic research data
data = {
    'participant_id': range(1, n_samples + 1),
    'age': np.random.normal(35, 10, n_samples).clip(18, 80),
    'income': np.random.lognormal(10, 0.5, n_samples),
    'education_years': np.random.normal(14, 3, n_samples).clip(8, 20),
    'satisfaction_score': np.random.uniform(1, 10, n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.35, 0.25]),
    'treatment_group': np.random.choice(['control', 'treatment'], n_samples),
    'outcome': np.random.normal(50, 15, n_samples)
}

# Add some correlations to make it more realistic
data['outcome'] = (data['satisfaction_score'] * 3 + 
                   data['education_years'] * 1.5 + 
                   np.random.normal(0, 10, n_samples))

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values to demonstrate preprocessing
missing_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[missing_indices, 'income'] = np.nan

print(f"Sample data created with shape: {df.shape}")
print("First 5 rows:")
print(df.head())

"""
## Data Loading and Exploration
"""

# Initialize components
loader = DataLoader('data')
preprocessor = DataPreprocessor()
analyzer = DataAnalyzer()
visualizer = Visualizer()

# Save sample data
loader.save_csv(df, 'sample_research_data.csv')

# Load data back (demonstrating loading functionality)
df_loaded = loader.load_csv('sample_research_data.csv')

print(f"Data loaded successfully. Shape: {df_loaded.shape}")

"""
## Descriptive Analysis
"""

# Generate descriptive statistics
desc_stats = analyzer.descriptive_statistics(df_loaded)

print("Basic Dataset Information:")
print(f"- Shape: {desc_stats['basic_info']['shape']}")
print(f"- Columns: {desc_stats['basic_info']['columns']}")
print(f"- Missing values: {sum(desc_stats['missing_values'].values())}")

"""
## Data Preprocessing
"""

# Clean and preprocess data
df_clean = preprocessor.clean_data(df_loaded)
df_processed = preprocessor.handle_missing_values(df_clean, strategy='mean')
df_encoded = preprocessor.encode_categorical(df_processed)

print(f"Preprocessing completed. Final shape: {df_encoded.shape}")
print("Preprocessing summary:", preprocessor.get_preprocessing_summary())

"""
## Statistical Analysis
"""

# Correlation analysis
corr_matrix = analyzer.correlation_analysis(df_encoded)
print("Correlation analysis completed")

# Statistical tests with outcome variable
if 'outcome' in df_encoded.columns:
    stat_tests = analyzer.statistical_tests(df_encoded, 'outcome')
    print(f"Statistical tests completed for {len(stat_tests)} variables")

# Quick model evaluation
model_results = analyzer.quick_model_evaluation(df_encoded, 'outcome')
print(f"Model evaluation completed. R² score: {model_results.get('r2', 'N/A'):.3f}")

"""
## Visualizations
"""

# Create comprehensive dashboard
dashboard = visualizer.create_dashboard(df_encoded, 'outcome')

print(f"Dashboard created with {len(dashboard)} visualization categories")

# Show individual plots
if 'distributions' in dashboard:
    print("Distribution plots created for numeric variables")

if 'correlation_matrix' in dashboard:
    print("Correlation matrix visualization created")

"""
## Results Summary
"""

analysis_summary = analyzer.get_analysis_summary()
print("Analysis Summary:")
print(f"- Analyses performed: {analysis_summary['analyses_performed']}")
print(f"- Models trained: {analysis_summary['models_trained']}")

print("\nDemo completed successfully! All components are working correctly.")

"""
## Next Steps

This demo showed basic functionality. For real research projects, you can:

1. Load your actual research data using the DataLoader
2. Apply appropriate preprocessing steps
3. Perform comprehensive statistical analysis
4. Generate publication-ready visualizations
5. Build and evaluate predictive models

Refer to the documentation for advanced features and customization options.
"""