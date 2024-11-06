import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from preprocessing import apply_complete_preprocessing
import os.path

# To get the path to data/ regardless of where this script is called from :

# Given that data is two directories up from this file, 
# we need to go up two directories from this file to get to the data directory
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

# We can then get the paths to the data files
TIMESERIES_PATH = os.path.join(DATA_PATH, 'df_timeseries_en.tsv')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'df_data_en_processed.tsv')
METADATA_HELPER_PATH = os.path.join(DATA_PATH, 'yt_metadata_helper.feather')

# The number of rows to load at once
CHUNK_SIZE = 1000

def load(path, usecols = None, nrows = None, index_col = None, verbose = False):
    """
    Load the data from the file at the given path

    Parameters:
    path (str): the path to the file
    usecols (list): the columns to load
    nrows (int): the number of rows to load
    verbose (bool): whether to print the progress
    Return:
    df (pd.DataFrame): the data
    """
    if verbose:
        nb_rows = sum(1 for _ in open(path))
        df = pd.concat([chunk for chunk in tqdm(pd.read_csv(path, sep='\t', chunksize=CHUNK_SIZE, usecols=usecols, nrows=nrows, index_col=index_col, engine='c'), desc=f'Loading data in chunks of {CHUNK_SIZE}', total=nb_rows/CHUNK_SIZE)])
        print(f'Loaded {len(df)} rows')
    else:
        df = pd.read_csv(path, sep='\t', usecols=usecols, nrows=nrows, index_col=index_col, engine='c')
    
    return df

def load_feather(path, usecols = None, nrows = None, index_col = None, verbose = False):
    """
    Load the data from the feather file at the given path

    Parameters:
    path (str): the path to the file
    usecols (list): the columns to load
    nrows (int): the number of rows to load
    verbose (bool): whether to print the progress
    Return:
    df (pd.DataFrame): the data
    """
    if verbose:
        print(f'Loading data from \'{path}\'...')
    df = pd.read_feather(path, columns=usecols)

    return df

def load_timeseries(usecols = None, nrows = None, verbose = False):
    """
    Load the time series data from the file

    Parameters:
    usecols (list): the columns to load
    nrows (int): the number of rows to load
    verbose (bool): whether to print the progress
    Return:
    df (pd.DataFrame): the time series data
    """
    return load(TIMESERIES_PATH, usecols=usecols, nrows=nrows, verbose=verbose)

def load_metadata_helper(usecols = None, nrows = None, verbose = False):
    """
    Load the metadata helper data from the file

    Parameters:
    usecols (list): the columns to load
    nrows (int): the number of rows to load
    verbose (bool): whether to print the progress
    Return:
    df (pd.DataFrame): the metadata helper data
    """
    return load_feather(METADATA_HELPER_PATH, usecols=usecols, nrows=nrows, verbose=verbose)

def load_processed_data(usecols = None, nrows = None, verbose = False):
    """
    Load the processed time series data from the file,
    saving the result in 'df_timeseries_en_processed.tsv'

    Parameters:
    usecols (list): the columns to load
    nrows (int): the number of rows to load
    verbose (bool): whether to print the progress
    Return:
    df (pd.DataFrame): the processed time series data
    """
    return load(PROCESSED_DATA_PATH, usecols=usecols, nrows=nrows, index_col=['channel', 'week'], verbose=verbose)

def update_processed_data(verbose = False):
    """
    Update the processed time series data (can take around 5 minutes)

    Parameters:
    verbose (bool): whether to print the progress
    """
    start_time = time()

    # Load the raw data
    data = load_timeseries(verbose=verbose)
    df_metadata_helper = load_metadata_helper(verbose=verbose)

    if verbose:
        print(f'Preprocessing...', end='\r')

    # Apply the preprocessing
    data = apply_complete_preprocessing(data, df_metadata_helper)

    if verbose:
        print('Preprocessing done:')
        print(data.head())

    # Save the processed data
    if verbose:
        chunks = np.array_split(data.index, 100) # split into 100 chunks
        for chunck, subset in enumerate(tqdm(chunks, desc='Saving data', total=len(chunks))):
            if chunck == 0: # first row
                data.loc[subset].to_csv(PROCESSED_DATA_PATH, mode='w', index=True, sep='\t')
            else:
                data.loc[subset].to_csv(PROCESSED_DATA_PATH, header=None, mode='a', index=True, sep='\t')
    else:
        data.to_csv(PROCESSED_DATA_PATH, sep='\t', index=True)

    if verbose:
        duration = time() - start_time
        print(f'Processed time series data updated in \'{PROCESSED_DATA_PATH}\' in {duration:.2f}s')

update_processed_data(verbose=True)
print(load_processed_data(verbose=True))