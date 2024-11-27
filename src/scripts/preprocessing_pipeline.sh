#!/bin/bash

set -e # Stop the script on any error

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src" # Add the src directory to the Python path

# Activate the virtual environment
echo "Activating the virtual environment"
source venv/bin/activate

pip install -r pip_requirements.txt

echo "Running the preprocessing pipeline"

echo "Perform data preprocessing"
python src/data/preprocessing.py # Run the data loader
echo "Perform BB data preprocessing"
python src/data/bbdataset_preprocessing.py # Run the data loader
# echo "Perform metadata preprocessing"
# python src/data/reduce_metadata.py # Run the metadata reducer

echo "Deactivating the virtual environment"
deactivate # Deactivate the virtual environment

echo "Preprocessing pipeline completed"