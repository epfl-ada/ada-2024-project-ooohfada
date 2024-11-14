#!/bin/bash

set -e # Stop the script on any error

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src" # Add the src directory to the Python path

source ../venv/bin/activate # Activate the virtual environment

echo "Running the preprocessing pipeline"

python data/preprocessing.py # Run the data loader
python data/bbdataset_preprocessing.py # Run the data loader
python data/reduce_metadata.py # Run the metadata reducer

deactivate # Deactivate the virtual environment