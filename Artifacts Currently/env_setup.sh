#!/bin/bash

# Enhanced Computer Vision Pipeline Environment Setup
echo "🚀 Setting up Enhanced CV Pipeline Environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'enhanced_cctv'..."
conda create -n enhanced_cctv python=3.9 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate enhanced_cctv

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Core computer vision libraries
echo "📷 Installing computer vision libraries..."
pip install ultralytics  # YOLOv8/YOLOv11
pip install opencv-python opencv-contrib-python

# Tracking libraries
echo "🎬 Installing tracking libraries..."
pip install motmetrics filterpy

# ML/AI libraries
echo "🤖 Installing ML/AI libraries..."
pip install transformers sentence-transformers
pip install open-clip-torch
pip install timm  # For additional backbone models

# Advanced training optimization
echo "⚡ Installing optimization libraries..."
pip install optuna  # Hyperparameter optimization
pip install wandb   # Experiment tracking
pip install ray[tune]  # Distributed hyperparameter tuning

# Data handling and visualization
echo "📊 Installing data and visualization libraries..."
pip install pandas numpy matplotlib seaborn plotly
pip install albumentations  # Advanced augmentations
pip install imgaug  # Additional augmentations

# Attribute classification specific
echo "🎯 Installing classification libraries..."
pip install scikit-learn
pip install xgboost lightgbm  # For ensemble methods

# Utility libraries
echo "🛠️ Installing utility libraries..."
pip install tqdm rich typer  # Better CLI interfaces
pip install pydantic  # Data validation
pip install hydra-core  # Configuration management

# Jupyter and development tools
echo "📝 Installing development tools..."
pip install jupyter ipywidgets
pip install black flake8 mypy  # Code formatting and type checking

# Install additional dependencies
echo "🔧 Installing additional dependencies..."
pip install efficientnet-pytorch  # For attribute classification backbones
pip install pretrainedmodels
pip install pytorch-lightning  # For structured training

# Verification
echo "✅ Verifying installations..."

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || echo "❌ PyTorch installation failed"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" || echo "❌ OpenCV installation failed"
python -c "import optuna; print(f'Optuna: {optuna.__version__}')" || echo "❌ Optuna installation failed"
python -c "import ultralytics; print('YOLOv8: OK')" || echo "❌ YOLOv8 installation failed"

echo "🎯 Environment setup complete!"
echo "📁 To activate the environment, run: conda activate enhanced_cctv"

# Create project directory structure
echo "📁 Creating project directory structure..."
mkdir -p {data/{raw_videos,attribute_dataset/images,models,output,logs},src,results,configs,tests}

echo "📂 Project structure created:"
echo "   data/raw_videos/        - Place your video files here"
echo "   data/attribute_dataset/ - Person attribute training data"
echo "   data/models/           - Trained models"
echo "   data/output/           - Processing results"
echo "   src/                   - Source code"
echo "   results/               - Experiment results"
echo "   configs/               - Configuration files"
echo "   tests/                 - Test scripts"

echo "🚀 Ready to start building your CCTV pipeline!"
