import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import os.path

# To get the path to data/ regardless of where this script is called from :

# Given that data is two directories up from this file, 
# Need to go up two directories from this file to get to the data directory
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

# We can then get the paths to the data files
TIMESERIES_PATH = os.path.join(DATA_PATH, 'df_timeseries_en.tsv')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'df_data_en_processed.tsv')
METADATA_HELPER_PATH = os.path.join(DATA_PATH, 'yt_metadata_helper.feather')
PROCESSED_BAD_BUZZ_PATH = os.path.join(DATA_PATH, 'df_bb_data_en_processed.tsv')

# The number of rows to load at once
CHUNK_SIZE = 1000

def load_raw_data(filename, n_first_rows):
    """
    Load the raw data from the given file and return the first n_first_rows
    Parameters:
    filename (str): the name of the file to load
    n_first_rows (int): the number of rows to load
    Return:
    df (pd.DataFrame): the first n_first_rows of the file
    """
    
    df_timeseries = pd.read_csv(f'./../../data/{filename}', sep='\t', compression='infer', nrows=n_first_rows)
    df_timeseries['datetime'] = pd.to_datetime(df_timeseries['datetime'])

    return df_timeseries


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

def load_bb_timeseries_processed(usecols = None, nrows = None, verbose = False):
    """
    Load the bad buzz df, preprocessed

    Args:
    usecols (list): the columns to load
    nrows (int): the number of rows to load
    verbose (bool): whether to print the progress

    Returns:
    df: the Timeseries df
    """
    return load(PROCESSED_BAD_BUZZ_PATH, usecols=usecols, nrows=nrows, index_col=['channel', 'week'], verbose=verbose)
