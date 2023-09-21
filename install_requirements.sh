#!/bin/bash

# Activate the virtual environment
echo "Activating virtual environment..."
. .venv/bin/activate

# Install pip
echo "Installing pip..."
python get-pip.py

# Install torch, torchvision, and torchaudio
echo "Installing pytorch..."
pip3 install torch torchvision torchaudio

# Install other requirements
echo "Installing other requirements..."
pip3 install dominate
pip3 install visdom

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo "Requirements installed successfully."
else
    echo "Failed to install requirements. Please check your setup."
fi
