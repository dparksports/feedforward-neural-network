#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

echo "Creating Python virtual environment in ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"

echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment successfully initialized!"
echo "To activate manually run: source .venv/bin/activate"
