import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import csv
from tqdm import tqdm

from src.data.dataloader_functions import *
from src.utils.results_utils import *

def load_data(file_path, verbose=True):
    """
    Load data from a file
    
    Parameters:
    file_path (str): Path to the file
    verbose (bool): Whether to print information on loading process
    """

    if verbose:
        print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path, sep='\t')
    df.set_index(['channel', 'week'], inplace=True)
    df['decline_event_detected'] = df['growth_diff'] < 0

    if verbose:
        print(f"Data loaded. \n")
    
    return df

def detect_decline_events(df_grouped, verbose=True):
    """
    Detect decline events in the dataset

    Parameters:
    df_grouped (pd.DataFrameGroupBy): Grouped dataframe containing the data for each channel
    verbose (bool): Whether to print information
    
    Returns:
    dict: Dictionary containing decline events with start and end week
    """

    if verbose:
        print("Detecting decline events ...")
    
    decline_events = {}
    for channel in tqdm(df_grouped.groups.keys(), desc="Processing channels"):
        channel_data = df_grouped.get_group(channel)
        for i in range(1, len(channel_data)):
            if channel_data['decline_event_detected'].iloc[i] and not channel_data['decline_event_detected'].iloc[i-1]:
                if channel not in decline_events:
                    decline_events[channel] = []
                decline_events[channel].append((channel_data['week'].iloc[i], None))
            if not channel_data['decline_event_detected'].iloc[i] and channel_data['decline_event_detected'].iloc[i-1]:
                decline_events[channel][-1] = (decline_events[channel][-1][0], channel_data['week'].iloc[i])

    if verbose:
        print(f"\nInitial decline events detected. \n")

    return decline_events

def filter_decline_events(decline_events, min_duration, verbose=True):
    """
    Filter decline events based on the minimum duration

    Parameters:
    decline_events (dict): Dictionary containing decline events
    min_duration (int): Minimum duration of the decline event
    verbose (bool): Whether to print information

    Returns:
    dict: Filtered decline events
    """

    if verbose:
        print(f"Filtering decline events with minimum duration of {min_duration} weeks ...")
    filtered_on_duration = {k: [x for x in v if x[1] >= min_duration] for k, v in decline_events.items()}

    if verbose:
        print(f"Decline events filtered. \n")
    
    return filtered_on_duration

def plot_duration_distribution(duration_list, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(duration_list, bins=20, kde=True)
    plt.title(title)
    plt.xlabel('Duration (weeks)')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.show()

def filter_on_growth_diff(df_grouped, decline_events, min_growth_diff_percentage, verbose=True):
    """
    Filter decline events based on the minimum growth difference percentage
    
    Parameters:
    df_grouped (pd.DataFrameGroupBy): Grouped dataframe
    decline_events (dict): Dictionary containing decline events
    min_growth_diff_percentage (int): Minimum growth difference percentage
    verbose (bool): Whether to print information
    
    Returns:
    dict: Filtered decline events
    """

    if verbose:
        print(f"Filtering decline events with minimum growth difference percentage of {min_growth_diff_percentage} ...")

    filtered_on_growth_diff = {}
    for channel in tqdm(df_grouped.groups.keys(), desc="Processing channels"):
        channel_data = df_grouped.get_group(channel)
        for event in decline_events.get(channel, []):
            start_week = event[0][0]
            end_week = event[0][1]
            min_growth_diff = channel_data.loc[(channel_data['week'] >= start_week) & (channel_data['week'] <= end_week)]['growth_diff_percentage'].min()
            if min_growth_diff < min_growth_diff_percentage:
                if channel not in filtered_on_growth_diff:
                    filtered_on_growth_diff[channel] = []
                filtered_on_growth_diff[channel].append(event)

    if verbose:
        print(f"Decline events filtered. \n")
    
    return filtered_on_growth_diff

def save_to_csv(file_path, data, headers, verbose=True):
    """
    Save data to a CSV file
    
    Parameters:
    file_path (str): Path to the file
    data (list): Data to save
    headers (list): Headers for the CSV file
    verbose (bool): Whether to print information
    """

    if verbose:
        print(f"Saving data to {file_path} ...")
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
    
    if verbose:
        print(f"Data saved. \n")

def create_decline_events_datasets(verbose=True):
    """
    Create the final datasets containing the decline events

    Parameters:
    verbose (bool): Whether to print information

    Returns:
    pd.DataFrame, pd.DataFrameGroupBy, dict: Dataframes for plotting purposes
    """

    if verbose:
        print("Starting decline events dataset creation: \n")

    df_with_rgr_new = load_data('data/df_with_rgr_new.tsv', verbose=verbose)
    df_with_rgr_grouped = df_with_rgr_new.reset_index().groupby('channel')
    if verbose:
        print(f'Total number of channels: {len(df_with_rgr_grouped.groups.keys())} \n')

    decline_events = detect_decline_events(df_with_rgr_grouped, verbose=verbose)

    # Remove events with no end week
    decline_events = {k: [x for x in v if x[1] is not None] for k, v in decline_events.items()}

    # Add duration
    for channel in decline_events:
        for i in range(len(decline_events[channel])):
            decline_events[channel][i] = ((decline_events[channel][i][0], decline_events[channel][i][1]), decline_events[channel][i][1] - decline_events[channel][i][0])

    # Analyze the duration of the decline events
    duration_list = [event[1] for channel in decline_events for event in decline_events[channel]]
    duration_list = np.array(duration_list)

    plot_duration_distribution(duration_list, 'Distribution of the duration of decline event in the inital dataset')

    print('We detect way too many decline events, so we need to find a way to filter them.\nWe therefore set a threshold of 8 weeks for the minimum duration of a decline event. \n')

    filtered_on_duration = filter_decline_events(decline_events, 8, verbose=verbose)

    duration_list_filtered = [event[1] for channel in filtered_on_duration for event in filtered_on_duration[channel]]
    duration_list_filtered = np.array(duration_list_filtered)

    plot_duration_distribution(duration_list_filtered, 'Distribution of the duration of decline events')

    print('\nAfter filtering:')
    print(f'    Mean duration of decline events: {np.mean(duration_list_filtered)}')
    print(f'    Median duration of decline events: {np.median(duration_list_filtered)}')
    print(f'    Min duration of decline events: {np.min(duration_list_filtered)}')
    print(f'    Max duration of decline events: {np.max(duration_list_filtered)} \n')

    # Compute the growth difference percentage
    df_with_rgr_final = copy.deepcopy(df_with_rgr_new)
    df_with_rgr_final['growth_diff_percentage'] = (df_with_rgr_final['growth_diff'] / df_with_rgr_final['rolling_growth_rate']) * 100
    df_with_rgr_grouped_final = df_with_rgr_final.reset_index().groupby('channel')

    # Filter on growth difference percentage
    filtered_on_growth_diff = filter_on_growth_diff(df_with_rgr_grouped_final, filtered_on_duration, -80, verbose=verbose)

    # Filter on duration on this new dataset to avoid outliers
    filtered_on_growth_diff_no_outliers = filter_decline_events(filtered_on_growth_diff, 8, verbose=verbose)

    print(f'There are {sum([len(v) for v in filtered_on_growth_diff_no_outliers.values()])} decline events (based on growth rate) with duration of at least 8 weeks. \n')

    # Merge the two filters
    decline_events_final = {k: filtered_on_growth_diff_no_outliers.get(k, []) for k in set(filtered_on_growth_diff_no_outliers)}
    decline_events_final_sorted = {k: sorted(v, key=lambda x: x[0][0]) for k, v in decline_events_final.items()}

    if verbose:
        print(f'Number of channels with decline events detected after filtering: {len(decline_events_final_sorted)} \n')

    # Saving the final decline events dataset to csv
    save_to_csv('data/decline_events_complete.csv', [(channel, event, duration) for channel, events in decline_events_final_sorted.items() for event, duration in events], ["Channel", "Event", "Duration"], verbose=verbose)

    if verbose:
        print('Decline events saved to data/decline_events_complete.csv \n')

    # Create the 1M Plus dataset
    create_1m_plus_dataset(df_with_rgr_new, decline_events_final_sorted, verbose=verbose)

    if verbose:
        print('Decline events dataset creation completed.')

    # Return the dataframes for plotting purposes
    return df_with_rgr_final, df_with_rgr_grouped_final, decline_events_final_sorted

def create_1m_plus_dataset(df_with_rgr, df_decline_events_final_sorted, verbose=True):
    """
    Create the 1M Plus dataset

    Parameters:
    df_with_rgr (pd.DataFrame): Dataframe containing the data
    df_decline_events_final_sorted (dict): Dictionary containing decline events
    verbose (bool): Whether to print information
    """

    if verbose:
        print('Creating dataset for 1M Plus dataset (only channels in detected decline events) with more than 1M subscribers ...')

    # Compute the mean number of subscribers for each channel and filter on channels with more than 1M subscribers
    mean_subscribers = df_with_rgr.reset_index().groupby('channel')['subs'].mean().reset_index()
    mean_subscribers.columns = ['Channel', 'Mean_Number_of_Subscribers']
    mean_subscribers_dict = mean_subscribers.set_index('Channel')['Mean_Number_of_Subscribers'].to_dict()
    filtered_mean_subscribers_dict = {k: v for k, v in mean_subscribers_dict.items() if v > 1e6}

    decline_events_1M_plus = df_decline_events_final_sorted.copy()
    for k in list(decline_events_1M_plus.keys()):
        if k in filtered_mean_subscribers_dict:
            decline_events_1M_plus[k].append(filtered_mean_subscribers_dict[k])
        else:
            del decline_events_1M_plus[k]

    # Write to CSV file
    with open('data/1M_plus_from_declined_events.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Channel", "Event", "Duration", "Mean_Number_of_Subscribers"])
        # Write rows
        for channel, events in decline_events_1M_plus.items():
            mean_subscribers = events[-1]  
            for event, end_week in events[:-1]:
                writer.writerow([channel, event, end_week, mean_subscribers])

    if verbose:
        print('1M Plus dataset created and saved to data/1M_plus_from_declined_events.csv \n')

if __name__ == '__main__':
    create_decline_events_datasets(verbose=True)
