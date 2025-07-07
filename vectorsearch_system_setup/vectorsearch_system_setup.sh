#!/bin/bash
# vectorsearch_system_setup.sh
# System setup script for VectorSearch on Ubuntu-based VMs (e.g., Azure Standard_E64ds_v4)

echo "=== Updating packages ==="
sudo apt-get update

echo "=== Installing system dependencies ==="
sudo apt-get install -y build-essential cmake python3-dev python3-pip git

echo "=== Installing Python packages ==="
pip3 install --upgrade pip
pip3 install numpy faiss-cpu

echo "=== Setup complete ==="
