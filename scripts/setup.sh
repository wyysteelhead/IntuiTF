#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p src/{core,utils,models,visualization,config}
mkdir -p data/{raw,processed}
mkdir -p docs/images
mkdir -p notebooks
mkdir -p examples
mkdir -p tests

# Install DiffDVR if not already installed
if [ ! -d "DiffDVR" ]; then
    echo "Installing DiffDVR..."
    git clone https://github.com/shamanDevel/DiffDVR.git
    cd DiffDVR
    # Follow DiffDVR installation instructions
    cd ..
fi

# Set up conda environment
conda create -n geneticdvr python=3.8.8 -y
conda activate geneticdvr

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set permissions for scripts
chmod +x scripts/*.sh

echo "Setup complete! Don't forget to activate the conda environment with:"
echo "conda activate geneticdvr"