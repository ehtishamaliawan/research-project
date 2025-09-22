#!/bin/bash

# Setup script for research project development environment

echo "Setting up Research Project development environment..."

# Check if Python 3.8+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python $python_version is compatible"
else
    echo "✗ Python 3.8+ required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy jupyter

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{raw,processed,external}
mkdir -p results
mkdir -p logs

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py

# Run tests to verify installation
echo "Running tests to verify installation..."
python -m pytest tests/ -v

echo "✓ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To get started:"
echo "1. Place your data files in data/raw/"
echo "2. Run: python scripts/process_data.py data/raw/your_data.csv"
echo "3. Or use the Python API as shown in the README"