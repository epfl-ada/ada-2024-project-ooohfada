import pandas as pd

def apply_complete_preprocessing(df_timeseries, df_metadata_helper):
    """
    Apply the complete preprocessing to the time series and metadata helper data

    Parameters:
    df_timeseries (pd.DataFrame): the time series data
    df_metadata_helper (pd.DataFrame): the metadata helper data
    Return:
    df_preprocessed_data (pd.DataFrame): the preprocessed data, containing the preprocessed time series and metadata helper data
    """
    
    # Apply the preprocessing to the time series and metadata helper data
    df_timeseries = apply_timeseries_preprocessing(df_timeseries)
    df_metadata_helper = apply_metadata_helper_preprocessing(df_metadata_helper)

    # Merge the time series and metadata helper data
    df_preprocessed_data = df_timeseries.merge(df_metadata_helper, how='left', left_on=['channel', 'week'], right_on=['channel', 'week'])

    # Set views, likes and dislikes to 0 if activity is 0
    df_preprocessed_data.loc[df_preprocessed_data['activity'] == 0, ['view_count', 'like_count', 'dislike_count']] = 0

    return df_preprocessed_data

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
    df_timeseries = map_column_to_week(df_timeseries, 'datetime')

    # Merge the rows with the same channel and week
    df_timeseries = merge_duplicate_rows(df_timeseries)

    # Make the channel id and the week index the index
    df_timeseries = df_timeseries.set_index(['channel', 'week'])
    df_timeseries = df_timeseries.sort_index()

    # Add the empirical deltas
    df_timeseries = add_empirical_deltas(df_timeseries)
    
    return df_timeseries

def apply_metadata_helper_preprocessing(df_metadata_helper):
    """
    Apply the complete preprocessing to the metadata data

    Steps:
    - Convert the datetime column to datetime format
    - Map the week index to the date

    Parameters:
    df_metadata (pd.DataFrame): the metadata data
    Return:
    df_metadata (pd.DataFrame): the preprocessed metadata data
    """

    # Change the name of column 'channel_id' to 'channel', to match the timeseries data (later merging purpose)
    df_metadata_helper = df_metadata_helper.rename(columns={'channel_id': 'channel'})
    
    # Convert the upload date column to datetime format
    df_metadata_helper['upload_date'] = pd.to_datetime(df_metadata_helper['upload_date'])
    
    # Map the upload date to the week index
    df_metadata_helper = map_column_to_week(df_metadata_helper, 'upload_date')

    df_metadata_helper = count_views_likes_dislikes_per_week(df_metadata_helper)

    # Make the channel id and the week index the index
    df_metadata_helper = df_metadata_helper.set_index(['channel', 'week'])
    df_metadata_helper = df_metadata_helper.sort_index()
    
    return df_metadata_helper

def map_column_to_week(df, column_name):
    """
    Replace the given column by a week index,
    starting from 0 in the earliest week found in the dataset

    Parameters:
    df  (pd.DataFrame): the dataframe in which to replace the week index
    column_name (str): the column to replace (must be a date: 'datetime' for timeseries, 'upload_date' for metadata)
    Return:
    df_week (pd.DataFrame): the dataframe with the week index
    """

    # Get the first date in the dataset
    if column_name == 'upload_date':
        # The first date in the metadata is 2015-01-05, handle the time lag between the first date in metadata_helper and timeseries => keep only the data from 2015-01-05 (aligned with timeseries)
        first_date = pd.to_datetime('2015-01-05 00:00:00')

        # Drop all raws with upload_date before 2015-01-05
        df = df[df['upload_date'] >= first_date]
    else:
        first_date = df[column_name].min()

    print(first_date)

    # Compute the week index
    df['week'] = df[column_name].apply(lambda x: (x - first_date).days // 7)

    df_week = df.drop(column_name, axis=1)

    # Remove the datetime column
    return df_week

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

def count_views_likes_dislikes_per_week(df_metadata_helper):
    """
    Count the number of views, likes and dislikes for videos uploaded in each week

    Parameters:
    df_metadata (pd.DataFrame): the metadata_helper dataframe (columns: channel, week, view_count, like_count, dislike_count, categories, crawl_date, display_id, duration)
    Return:
    df_likes_dislikes (pd.DataFrame): the data frame with number of views, likes and dislikes for videos uploaded in each week
    """

    # Keep only the columns of interest
    df_metadata_helper = df_metadata_helper[['channel', 'week', 'view_count', 'like_count', 'dislike_count']]
    
    # Count the number of likes and dislikes per week
    df_likes_dislikes = df_metadata_helper.groupby(['channel', 'week']).agg({
        'view_count': 'sum',
        'like_count': 'sum',
        'dislike_count': 'sum',
    }).reset_index()
    
    return df_likes_dislikes

# Some tests to check the preprocessing
if __name__ == '__main__':
    timeseries_example = pd.DataFrame({
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

    metadata_helper_example = pd.DataFrame({
        'upload_date': ['2020-01-01', '2020-01-01', '2020-01-08', '2020-01-08'],
        'channel': ['A', 'A', 'A', 'A'],
        'view_count': [100, 200, 320, 430],
        'like_count': [10, 20, 30, 40],
        'dislike_count': [1, 5, 10, 20],
        'categories': ['Music', 'Music', 'Music', 'Music'],
        'crawl_date': ['2020-01-01', '2020-01-01', '2020-01-08', '2020-01-08'],
        'display_id': ['1', '2', '3', '4'],
        'duration': [1, 2, 3, 5],
    })

    print("Before preprocessing:")
    print('\nTimeseries:')
    print(timeseries_example)
    print('\nMetadata helper:')
    print(metadata_helper_example)

    print("\nAfter preprocessing:")
    print('\nTimeseries:')
    print(apply_timeseries_preprocessing(timeseries_example))
    print('\nMetadata helper:')
    print(apply_metadata_helper_preprocessing(metadata_helper_example))
    print('\nComplete preprocessing:')
    print(apply_complete_preprocessing(timeseries_example, metadata_helper_example))
    