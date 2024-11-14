#!/bin/bash

set -e # Stop the script on any error

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src" # Add the src directory to the Python path

echo "Activating the virtual environment"
source ../../venv/bin/activate # Activate the virtual environment

echo "Running the preprocessing pipeline"

echo "Perform data preprocessing"
python data/preprocessing.py # Run the data loader
echo "Perform BB data preprocessing"
python data/bbdataset_preprocessing.py # Run the data loader
echo "Perform metadata preprocessing"
python data/reduce_metadata.py # Run the metadata reducer

echo "Deactivating the virtual environment"
deactivate # Deactivate the virtual environment

echo "Preprocessing pipeline completed"