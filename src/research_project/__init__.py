"""
Research Project Package

A comprehensive Python package for research data analysis and machine learning.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .analyzer import DataAnalyzer
from .visualizer import Visualizer

__all__ = [
    "DataLoader",
    "DataPreprocessor", 
    "DataAnalyzer",
    "Visualizer"
]