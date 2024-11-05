import pandas as pd

def apply_timeseries_preprocessing(df_timeseries):
    """
    Apply the complete preprocessing to the time series data

    Steps:
    - Convert the datetime column to datetime format
    - Map the week index to the date

    Parameters:
    df_timeseries (pd.DataFrame): the time series data
    n_first_rows (int): the number of rows to load
    Return:
    df_timeseries (pd.DataFrame): the preprocessed time series data
    """
    
    # Convert the datetime column to datetime format
    df_timeseries['datetime'] = pd.to_datetime(df_timeseries['datetime'])
    
    # Map the datetime to the week index
    df_timeseries = map_datetime_to_week(df_timeseries)

    # Merge the rows with the same channel and week
    df_timeseries = merge_duplicate_rows(df_timeseries)

    # Make the channel id and the week index the index
    df_timeseries = df_timeseries.set_index(['channel', 'week'])
    df_timeseries = df_timeseries.sort_index()

    # Add the empirical deltas
    df_timeseries = add_empirical_deltas(df_timeseries)
    
    return df_timeseries

def map_datetime_to_week(df_timeseries):
    """
    Replace the datetime column by the week index,
    starting from 0 in the earliest week found in the dataset

    Parameters:
    df_timeseries (pd.DataFrame): the time series data
    Return:
    df_timeseries (pd.DataFrame): the time series data with the week index
    """
    
    # Get the first date in the dataset
    first_date = df_timeseries['datetime'].min()

    # Compute the week index
    df_timeseries['week'] = df_timeseries['datetime'].apply(lambda x: (x - first_date).days // 7)

    # Remove the datetime column
    return df_timeseries.drop('datetime', axis=1)

def add_empirical_deltas(df_timeseries):
    """
    Add the empirical sub, view and video deltas to the time series data

    Parameters:
    df_timeseries (pd.DataFrame): the time series data
    Return:
    df_timeseries (pd.DataFrame): the time series data with the empirical delta
    """

    # Compute the empirical delta, grouping by channel to avoid mixing channels
    df_timeseries['delta_subs'] = df_timeseries.groupby('channel')['subs'].diff()
    df_timeseries['delta_views'] = df_timeseries.groupby('channel')['views'].diff()
    df_timeseries['delta_videos'] = df_timeseries.groupby('channel')['videos'].diff()

    return df_timeseries

def merge_duplicate_rows(df_timeseries):
    """
    Merge the rows with the same channel and week by aggregating the values

    Parameters:
    df_timeseries (pd.DataFrame): the time series data
    Return:
    df_timeseries (pd.DataFrame): the time series data with the merged rows
    """

    # Group by channel and week then sum or take the last value depending on the column
    df_timeseries = df_timeseries.groupby(['channel', 'week']).agg({
        'category': 'last',
        'views': 'last',
        'delta_views': 'sum',
        'subs': 'last',
        'delta_subs': 'sum',
        'videos': 'last',
        'delta_videos': 'sum',
        'activity': 'last',
    }).reset_index()
    
    return df_timeseries

# Some tests to check the preprocessing
if __name__ == '__main__':
    example = pd.DataFrame({
        'datetime': ['2020-01-01', '2020-01-01', '2020-01-08', '2020-01-08'],
        'channel': ['A', 'A', 'A', 'A'],
        'category': ['Music', 'Music', 'Music', 'Music'],
        'views': [100, 200, 320, 430],
        'delta_views': [0, 0, 0, 0],
        'subs': [10, 28, 40, 38],
        'delta_subs': [0, 0, 0, 0],
        'videos': [1, 2, 3, 5],
        'delta_videos': [0, 1, 1, 2],
        'activity': [1, 2, 3, 4],
    })

    print("Before preprocessing:")
    print(example)

    print("\nAfter preprocessing:")
    print(apply_timeseries_preprocessing(example))