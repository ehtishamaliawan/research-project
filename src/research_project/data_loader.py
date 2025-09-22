"""
Data loading utilities for research project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A versatile data loader for various file formats commonly used in research.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initialize DataLoader with data directory.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataLoader initialized with directory: {self.data_dir}")
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame containing the loaded data
        """
        filepath = self.data_dir / filename
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def load_excel(self, filename: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            filename: Name of the Excel file
            sheet_name: Name of the sheet to load
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            DataFrame containing the loaded data
        """
        filepath = self.data_dir / filename
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def save_csv(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the output CSV file
            **kwargs: Additional arguments for df.to_csv
        """
        filepath = self.data_dir / filename
        try:
            df.to_csv(filepath, index=False, **kwargs)
            logger.info(f"Successfully saved {len(df)} rows to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to {filepath}: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about available data files.
        
        Returns:
            Dictionary containing file information
        """
        files_info = {}
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.csv', '.xlsx', '.json']:
                files_info[file_path.name] = {
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'type': file_path.suffix
                }
        return files_info