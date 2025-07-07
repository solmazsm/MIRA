#!/bin/bash
# azure_setup_vectorsearch.sh
# Setup script for VectorSearch on an Azure Ubuntu VM (Standard_E64ds_v4)

echo "=== [1/5] System Update ==="
sudo apt-get update && sudo apt-get upgrade -y

echo "=== [2/5] Install C++ Build Tools ==="
sudo apt-get install -y build-essential cmake g++ make libomp-dev

echo "=== [3/5] Install Python + PIP ==="
sudo apt-get install -y python3 python3-pip python3-venv python3-dev

echo "=== [4/5] Install Python Dependencies ==="
pip3 install --upgrade pip
pip3 install numpy faiss-cpu sentence-transformers

echo "=== [5/5] Setup Complete ==="
echo " VectorSearch environment is ready on Azure VM."
