# Research Project

A comprehensive Python package for research data analysis, statistical testing, and machine learning workflows. This project provides a complete toolkit for researchers to efficiently process, analyze, and visualize their data.

## Features

### 🔍 Data Loading and Processing
- **Multi-format support**: CSV, Excel, JSON data loading
- **Automatic data cleaning**: Handle missing values, duplicates, and inconsistencies  
- **Smart preprocessing**: Feature scaling, encoding, and transformation
- **Data validation**: Comprehensive data quality checks

### 📊 Statistical Analysis
- **Descriptive statistics**: Comprehensive summary statistics and data profiling
- **Correlation analysis**: Pearson, Spearman, and Kendall correlation matrices
- **Statistical testing**: T-tests, ANOVA, chi-square tests
- **Outlier detection**: IQR and Z-score based outlier identification

### 🤖 Machine Learning
- **Quick model evaluation**: Automated model selection and evaluation
- **Feature engineering**: Automatic feature creation and selection
- **Model comparison**: Compare multiple algorithms with cross-validation
- **Predictive analytics**: Both regression and classification support

### 📈 Data Visualization
- **Interactive plots**: Plotly-based interactive visualizations
- **Statistical charts**: Distribution plots, correlation heatmaps, box plots
- **Dashboard creation**: Automated comprehensive visualization dashboards
- **Publication-ready plots**: High-quality matplotlib figures

## Installation

```bash
# Clone the repository
git clone https://github.com/ehtishamaliawan/research-project.git
cd research-project

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from research_project import DataLoader, DataPreprocessor, DataAnalyzer, Visualizer

# Initialize components
loader = DataLoader('data')
preprocessor = DataPreprocessor()
analyzer = DataAnalyzer()
visualizer = Visualizer()

# Load your data
df = loader.load_csv('your_data.csv')

# Preprocess data
df_clean = preprocessor.clean_data(df)
df_processed = preprocessor.handle_missing_values(df_clean)
df_encoded = preprocessor.encode_categorical(df_processed)

# Analyze data
desc_stats = analyzer.descriptive_statistics(df_encoded)
correlation = analyzer.correlation_analysis(df_encoded)

# Create visualizations
dashboard = visualizer.create_dashboard(df_encoded, target_col='your_target')
```

### Command Line Interface

Process data directly from the command line:

```bash
# Basic processing
python scripts/process_data.py data/your_data.csv --output-dir results/

# With target analysis
python scripts/process_data.py data/your_data.csv --target-column outcome --verbose
```

## Project Structure

```
research-project/
├── src/research_project/          # Main package source code
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessor.py           # Data preprocessing tools
│   ├── analyzer.py               # Statistical analysis functions
│   └── visualizer.py             # Visualization utilities
├── scripts/                      # Utility scripts
│   └── process_data.py          # Data processing pipeline
├── notebooks/                    # Jupyter notebooks and examples
│   └── research_demo.py         # Demo notebook (Python format)
├── tests/                        # Unit tests
│   └── test_research_project.py # Test suite
├── config/                       # Configuration files
│   └── default.json             # Default configuration
├── data/                         # Data directories
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data
│   └── external/                # External data sources
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── setup.py                     # Package setup file
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Documentation

### DataLoader Class
Handles loading data from various file formats with robust error handling.

**Methods:**
- `load_csv(filename, **kwargs)`: Load CSV files
- `load_excel(filename, sheet_name=None, **kwargs)`: Load Excel files
- `save_csv(df, filename, **kwargs)`: Save DataFrames to CSV
- `get_data_info()`: Get information about available data files

### DataPreprocessor Class
Comprehensive data preprocessing and cleaning utilities.

**Methods:**
- `clean_data(df, drop_duplicates=True)`: Basic data cleaning
- `handle_missing_values(df, strategy='mean')`: Missing value imputation
- `encode_categorical(df, method='label')`: Categorical variable encoding
- `scale_features(df, method='standard')`: Feature scaling and normalization
- `create_features(df)`: Automated feature engineering

### DataAnalyzer Class
Statistical analysis and model evaluation tools.

**Methods:**
- `descriptive_statistics(df)`: Generate comprehensive statistics
- `correlation_analysis(df, method='pearson')`: Correlation analysis
- `statistical_tests(df, target_col)`: Perform statistical tests
- `quick_model_evaluation(df, target_col)`: Automated model evaluation
- `outlier_detection(df, method='iqr')`: Identify outliers

### Visualizer Class
Create publication-ready visualizations and interactive dashboards.

**Methods:**
- `plot_distribution(df, column)`: Distribution plots
- `plot_correlation_matrix(df)`: Correlation heatmaps
- `plot_categorical_analysis(df, column)`: Categorical variable analysis
- `plot_time_series(df, date_col, value_cols)`: Time series visualization
- `create_dashboard(df, target_col)`: Comprehensive visualization dashboard

## Examples

### Example 1: Basic Data Analysis

```python
import pandas as pd
from research_project import DataAnalyzer

# Load your data
df = pd.read_csv('research_data.csv')

# Initialize analyzer
analyzer = DataAnalyzer()

# Get comprehensive statistics
stats = analyzer.descriptive_statistics(df)
print(f"Dataset has {stats['basic_info']['shape'][0]} rows and {stats['basic_info']['shape'][1]} columns")

# Analyze correlations
corr_matrix = analyzer.correlation_analysis(df)
```

### Example 2: Machine Learning Pipeline

```python
from research_project import DataPreprocessor, DataAnalyzer

# Initialize components
preprocessor = DataPreprocessor()
analyzer = DataAnalyzer()

# Preprocess data
df_clean = preprocessor.clean_data(df)
df_processed = preprocessor.handle_missing_values(df_clean)
df_ready = preprocessor.encode_categorical(df_processed)

# Quick model evaluation
results = analyzer.quick_model_evaluation(df_ready, 'target_variable')
print(f"Model R² Score: {results.get('r2', 'N/A')}")
```

### Example 3: Visualization Dashboard

```python
from research_project import Visualizer
import matplotlib.pyplot as plt

# Create visualizer
viz = Visualizer()

# Generate comprehensive dashboard
dashboard = viz.create_dashboard(df, target_col='outcome')

# Save all plots
for plot_name, figure in dashboard.items():
    if hasattr(figure, 'savefig'):
        figure.savefig(f'{plot_name}.png', dpi=300, bbox_inches='tight')
```

## Configuration

The project uses JSON configuration files in the `config/` directory. You can customize:

- Data paths and file locations
- Preprocessing parameters (scaling methods, imputation strategies)
- Analysis settings (correlation thresholds, significance levels)
- Visualization preferences (figure sizes, color schemes)
- Model parameters (estimators, cross-validation settings)

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_research_project.py -v

# Run with coverage report
python -m pytest tests/ --cov=research_project --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions, please:

1. Check the existing issues on GitHub
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about your environment and the issue

## Changelog

### Version 0.1.0
- Initial release with core functionality
- Data loading and preprocessing utilities
- Statistical analysis tools
- Visualization dashboard
- Command-line interface
- Comprehensive test suite