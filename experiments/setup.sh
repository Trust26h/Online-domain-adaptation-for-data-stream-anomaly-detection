#!/bin/bash

# MROT Experiment Framework - Quick Start Installation
# =====================================================

echo "=================================================="
echo "MROT Experiment Framework - Quick Start"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $python_version"

# Check for required packages
echo ""
echo "Checking required packages..."

packages=(
    "pandas"
    "numpy"
    "matplotlib"
    "seaborn"
    "scikit-learn"
    "pyyaml"
    "jupyter"
)

missing_packages=()

for package in "${packages[@]}"; do
    python3 -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        missing_packages+=("$package")
        echo "  ✗ $package - NOT FOUND"
    else
        echo "  ✓ $package - OK"
    fi
done

# Install missing packages
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo ""
    echo "Installing missing packages..."
    pip install --break-system-packages ${missing_packages[@]}
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Installation successful!"
    else
        echo "  ✗ Installation failed. Please install manually:"
        echo "    pip install ${missing_packages[@]}"
        exit 1
    fi
else
    echo ""
    echo "All required packages are installed!"
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p datasets/cao2025
mkdir -p results
mkdir -p source

echo "  ✓ Created directories"

# Check for source files
echo ""
echo "Checking source files..."

required_files=(
    "source/onlineMROTauc_eval.py"
    "source/onlineMROTwassertein.py"
    "source/offline.py"
    "source/utils.py"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file - FOUND"
    else
        echo "  ✗ $file - NOT FOUND"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Some source files are missing!"
    echo "Please ensure the following files are in the source/ directory:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
fi

# Check for datasets
echo ""
echo "Checking datasets..."

if [ -z "$(ls -A datasets/cao2025 2>/dev/null)" ]; then
    echo "  ⚠ No datasets found in datasets/cao2025/"
    echo "  Please add your CSV datasets to this directory"
else
    echo "  ✓ Datasets found:"
    ls datasets/cao2025/*.csv 2>/dev/null | sed 's/^/    /'
fi

# Final instructions
echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Quick Start Options:"
echo ""
echo "1. Python Script:"
echo "   python run_experiments.py"
echo ""
echo "2. Configuration File:"
echo "   Edit config.yaml, then run:"
echo "   python run_from_config.py"
echo ""
echo "3. Jupyter Notebook (Recommended):"
echo "   jupyter notebook multi_dataset_experiments.ipynb"
echo ""
echo "4. Compare Results:"
echo "   python compare_results.py"
echo ""
echo "Documentation:"
echo "  See README.md for detailed instructions"
echo ""
echo "=================================================="
