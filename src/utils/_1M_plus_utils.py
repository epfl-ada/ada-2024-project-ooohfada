import pandas as pd
from tqdm import tqdm
from src.utils.recovery_analysis_utils import *

METADATA_FILENAME = 'data/yt_metadata_en.jsonl'
TO_FILE = 'data/1M_plus_videos_title_around_declines.csv'

def extract_videos_title(df_decline_events, chunk_size=5000, verbose=True):
    """
    Extract the title, channel id and upload week of the videos that are in the time frame for the respective channel.
    The result is saved in a csv file `videos_title_around_declines.csv` under the data folder.

    Parameters:
    df_event (pd.DataFrame): the dataframe with the decline events
    chunk_size (int): the size of the chunks to read the metadata file
    """

    if verbose:
        print('Extracting videos title around declines... \n')

    pd.options.mode.chained_assignment = None # remove SettingWithCopyWarning: 

    channels = df_decline_events['Channel'].unique()

    i = 0
    for chunk in pd.read_json(METADATA_FILENAME, lines=True, chunksize=chunk_size):

        try:
            init_shape = int(chunk.shape[0])
            
            chunk = chunk[chunk['channel_id'].isin(channels)]
            
            chunk.loc[:, 'upload_date'] = pd.to_datetime(chunk['upload_date'])

            chunk = map_column_to_week(chunk, 'upload_date')

            # keep the videos that are in the time frame for the respective channel
            mask = []
            for video in chunk.itertuples():
                channel_mask = df_decline_events['Channel'].isin([video.channel_id])
                start_mask = df_decline_events['Start'] - df_decline_events['Duration'] <= video.week
                end_mask = df_decline_events['End'] >= video.week

                mask.append(df_decline_events[channel_mask & start_mask & end_mask].shape[0] > 0)

            kept = chunk[mask]

            if kept.shape[0] == 0:
                print(f'Chunk {i} (lines {i*chunk_size} to {(i+1)*chunk_size}): 0/{init_shape} videos kept')
                i += 1
                continue

            cols_of_interest = ['channel_id', 'week', 'title']

            kept = kept[cols_of_interest]

            print(f'Chunk {i} (lines {i*chunk_size} to {(i+1)*chunk_size}): {kept.shape[0]}/{init_shape} videos kept')

            kept.to_csv(TO_FILE, index=True, mode='a', header=False) # Temporary filename

            i += 1
        
        except Exception as e:
            print(f'Error in chunk {i} (lines {i*chunk_size} to {(i+1)*chunk_size}): {e}')

    if verbose:
        print('\nExtraction done.\n')

    if verbose:
        print('Saving the final file...')

    extracted_videos = pd.read_csv(TO_FILE)
    extracted_videos.reset_index(drop=False, inplace=True)
    if('level_0' in extracted_videos.columns):
        extracted_videos.drop(columns=['level_0'], inplace=True)
    extracted_videos.set_index([extracted_videos.columns[1], extracted_videos.columns[2]], inplace=True)
    extracted_videos.index.names = ['channel', 'week']

    if verbose:
        print('File saved.\n')


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

    # The first date in the metadata is 2015-01-05, handle the time lag between the first date in metadata_helper and timeseries => keep only the data from 2015-01-05 (aligned with timeseries)
    first_date = pd.to_datetime('2015-01-05 00:00:00')

    # Get the first date in the dataset
    if column_name == 'upload_date':
        # Drop all raws with upload_date before 2015-01-05
        df = df[df['upload_date'] >= first_date]

    # Compute the week index
    df['week'] = df[column_name].apply(lambda x: (x - first_date).days // 7)

    df_week = df.drop(column_name, axis=1)

    # Remove the datetime column
    return df_week


def count_values(df_to_count, columns):
    """
    Count the number of unique values in a column of a DataFrame.
    """
    value_counts_dict = {}

    for col in columns:
        value_counts = df_to_count[col].value_counts().to_dict()
        value_counts_dict[col] = value_counts

    df = pd.DataFrame(value_counts_dict).fillna(0).astype(int)
    df.index.name = "Value"
    df.columns.name = "Strategy"

    return df

def find_videos_after(df_videos, row):
    channel_mask = df_videos['channel'] == row['Channel']
    left_mask = df_videos['week'] >= row['Start']
    right_mask = df_videos['week'] <= row['End'] + row['Duration']

    return df_videos[channel_mask & left_mask & right_mask]['index'].tolist()

def match_declines_with_videos_after(df_declines, df_videos, verbose=True):
    """
    Match declines with videos uploaded after the decline.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the decline events
    df_videos (pd.DataFrame): The DataFrame containing the video metadata
    verbose (bool): Whether to print the progress

    Returns:
    pd.DataFrame: The decline events with the indices of videos uploaded after the decline
    """

    if verbose:
        print("Matching declines with videos after ... ")

    df_declines['Videos_after'] = [[]] * len(df_declines)

    for idx, row in tqdm(df_declines.iterrows(), total=len(df_declines)):
        df_declines.at[idx, 'Videos_after'] = find_videos_after(df_videos, row)

    if verbose:
        print("Declines matched with videos after.\n")

    return df_declines

def get_stats_from_declines(df_declines, df_channels, df_data_processed):
    """
    Get the stats from the decline events.
    
    Parameters:
    df_declines (pd.DataFrame): the decline events
    df_channels (pd.DataFrame): the channels data
    df_data_processed (pd.DataFrame): the processed data
    
    Returns:
    pd.DataFrame: the decline events with the stats
    """

    # If the decline is longer than 4 months without recovery, we consider the YouTuber was not successful in handling it.
    # Our aim is to find strategies that lead to quick recoveries, therefore taking more than 4 months (= 16 weeks) would be considered unsuccessful.
    RECOVERY_THRESHOLD = 16

    # Add the decline outcome
    df_declines['Recovered'] = df_declines['Duration'] < RECOVERY_THRESHOLD

    # Split the tuple (decline start, decline end) into two separate columns
    df_declines['Event'] = df_declines['Event'].apply(lambda s: [int(week_id) for week_id in s[1:-1].split(', ')])
    df_declines['Start'] = df_declines['Event'].apply(lambda e: e[0])
    df_declines['End'] = df_declines['Event'].apply(lambda e: e[1])
    df_declines.drop('Event', axis=1, inplace=True)

    # Add the channel category
    df_declines['Category'] = df_declines['Channel'].apply(lambda c: df_channels.loc[c]['category_cc'])

    # Add the channel's subs at the start of the decline
    decline_index = list(zip(df_declines['Channel'], df_declines['Start']))
    df_declines['Subs_start'] = df_data_processed.loc[decline_index, 'subs'].values

    # Add the activity at the start of the decline
    df_declines['Activity_start'] = df_data_processed.loc[decline_index, 'activity'].values

    # Add the channel's subs at the start of the decline
    df_declines['Views_start'] = df_data_processed.loc[decline_index, 'views'].values

    return df_declines

def propensity_score_matching(df, treatments, to_drop):
    for treatment, dropped in zip(treatments, to_drop):
        matches = get_matches(treatment=treatment, declines=df.drop(dropped, axis=1), verbose=False)
        # Flatten
        matches = [index for match in matches for index in match]

        # Get the matched declines
        df_matched = df.loc[matches]

        plot_treatment_effect(df_matched, treatment)

def get_combination_strategies(combination):
    """
    Extracts the strategies from a combination string.
    
    Parameters:
    - combination (str): Combination string of strategies separated by '_&_'
    
    Returns:
    - combination_strategies (list): List of strategies in the combination
    """

    # Extract the strategies from a combination 'strategy1_&_strategy2_&_...'
    combination_strategies = combination.split('_&_')

    return combination_strategies
