#!/bin/bash
# Create Python virtual environment and install requirements

set -e

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment setup complete. Activate with: source venv/bin/activate"
