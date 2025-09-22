#!/usr/bin/env python3
"""
Data processing utility script for research project.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_project import DataLoader, DataPreprocessor, DataAnalyzer, Visualizer


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process research data')
    parser.add_argument('input_file', help='Input data file (CSV or Excel)')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--target-column', help='Target column for analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data processing pipeline")
    
    try:
        # Initialize components
        loader = DataLoader('data')
        preprocessor = DataPreprocessor()
        analyzer = DataAnalyzer()
        visualizer = Visualizer()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {args.input_file}")
        if args.input_file.endswith('.csv'):
            df = loader.load_csv(args.input_file)
        elif args.input_file.endswith('.xlsx'):
            df = loader.load_excel(args.input_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Basic preprocessing
        logger.info("Starting data preprocessing")
        df_clean = preprocessor.clean_data(df)
        df_processed = preprocessor.handle_missing_values(df_clean)
        
        # Save processed data
        processed_file = output_dir / f"processed_{Path(args.input_file).name}"
        loader.data_dir = output_dir
        loader.save_csv(df_processed, processed_file.name)
        
        # Descriptive analysis
        logger.info("Performing descriptive analysis")
        desc_stats = analyzer.descriptive_statistics(df_processed)
        
        # Correlation analysis
        corr_matrix = analyzer.correlation_analysis(df_processed)
        
        # Target-based analysis if target column specified
        if args.target_column:
            logger.info(f"Performing target-based analysis for {args.target_column}")
            statistical_tests = analyzer.statistical_tests(df_processed, args.target_column)
            model_eval = analyzer.quick_model_evaluation(df_processed, args.target_column)
        
        # Outlier detection
        outliers = analyzer.outlier_detection(df_processed)
        
        # Generate visualizations
        logger.info("Creating visualizations")
        dashboard = visualizer.create_dashboard(df_processed, args.target_column)
        
        # Save plots
        for plot_name, fig in dashboard.items():
            if hasattr(fig, 'savefig'):  # matplotlib figure
                fig.savefig(output_dir / f"{plot_name}.png", dpi=300, bbox_inches='tight')
            elif hasattr(fig, 'items'):  # dictionary of figures
                for sub_name, sub_fig in fig.items():
                    if hasattr(sub_fig, 'savefig'):
                        sub_fig.savefig(output_dir / f"{plot_name}_{sub_name}.png", 
                                       dpi=300, bbox_inches='tight')
        
        # Generate summary report
        summary_report = {
            'data_info': desc_stats,
            'preprocessing_summary': preprocessor.get_preprocessing_summary(),
            'analysis_summary': analyzer.get_analysis_summary(),
        }
        
        # Save summary as text file
        with open(output_dir / 'analysis_summary.txt', 'w') as f:
            f.write("Research Data Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset shape: {df_processed.shape}\n")
            f.write(f"Number of numeric columns: {len(df_processed.select_dtypes(include=['number']).columns)}\n")
            f.write(f"Number of categorical columns: {len(df_processed.select_dtypes(include=['object']).columns)}\n")
            f.write(f"Missing values handled: {len(preprocessor.imputers)}\n")
            f.write(f"Analyses performed: {', '.join(analyzer.analysis_results.keys())}\n")
        
        logger.info(f"Processing completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()