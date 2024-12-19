import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import csv
from tqdm import tqdm

from src.data.dataloader_functions import *
from src.utils.results_utils import *

def load_data(file_path, verbose=False):
    if verbose:
        print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    df.set_index(['channel', 'week'], inplace=True)
    df['decline_event_detected'] = df['growth_diff'] < 0
    return df

def detect_decline_events(df_grouped, verbose=False):
    if verbose:
        print("Detecting decline events")
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
    return decline_events

def filter_decline_events(decline_events, min_duration, verbose=False):
    if verbose:
        print(f"Filtering decline events with minimum duration of {min_duration} weeks")
    filtered_on_duration = {k: [x for x in v if x[1] >= min_duration] for k, v in decline_events.items()}
    return filtered_on_duration

def plot_duration_distribution(duration_list, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(duration_list, bins=20, kde=True)
    plt.title(title)
    plt.xlabel('Duration (weeks)')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.show()

def filter_on_growth_diff(df_grouped, decline_events, min_growth_diff_percentage, verbose=False):
    if verbose:
        print(f"Filtering decline events with minimum growth difference percentage of {min_growth_diff_percentage}")
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
    return filtered_on_growth_diff

def save_to_csv(file_path, data, headers, verbose=False):
    if verbose:
        print(f"Saving data to {file_path}")
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)

def create_datasets(verbose=False):
    if verbose:
        print("Starting dataset creation")
    df_with_rgr_new = load_data('data/df_with_rgr_new.tsv', verbose=verbose)
    df_with_rgr_grouped = df_with_rgr_new.reset_index().groupby('channel')
    if verbose:
        print(f'Number of channels : {len(df_with_rgr_grouped.groups.keys())}')

    decline_events = detect_decline_events(df_with_rgr_grouped, verbose=verbose)

    # Add duration
    for channel in decline_events:
        for i in range(len(decline_events[channel])):
            decline_events[channel][i] = ((decline_events[channel][i][0], decline_events[channel][i][1]), decline_events[channel][i][1] - decline_events[channel][i][0])

    # Remove events with no end week
    decline_events = {k: [x for x in v if x[1] is not None] for k, v in decline_events.items()}

    duration_list = [event[1] for channel in decline_events for event in decline_events[channel]]
    duration_list = np.array(duration_list)

    print('Our initial dataset:')
    plot_duration_distribution(duration_list, 'Distribution of the duration of decline events')

    print('We detect way too many decline events, so we need to find a way to filter them.\nWe therefore set a threshold of 8 weeks for the minimum duration of a decline event.')

    filtered_on_duration = filter_decline_events(decline_events, 8, verbose=verbose)

    duration_list_filtered = [event[1] for channel in filtered_on_duration for event in filtered_on_duration[channel]]
    duration_list_filtered = np.array(duration_list_filtered)

    plot_duration_distribution(duration_list_filtered, 'Distribution of the duration of decline events')

    print('\nAfter filtering:')
    print(f'    Mean duration of decline events: {np.mean(duration_list_filtered)}')
    print(f'    Median duration of decline events: {np.median(duration_list_filtered)}')
    print(f'    Min duration of decline events: {np.min(duration_list_filtered)}')
    print(f'    Max duration of decline events: {np.max(duration_list_filtered)}')

    df_with_rgr_final = df_with_rgr_new
    df_with_rgr_final['growth_diff_percentage'] = (df_with_rgr_final['growth_diff'] / df_with_rgr_final['rolling_growth_rate']) * 100
    df_with_rgr_grouped_final = df_with_rgr_final.reset_index().groupby('channel')

    filtered_on_growth_diff = filter_on_growth_diff(df_with_rgr_grouped_final, filtered_on_duration, -80, verbose=verbose)
    decline_events_final = {k: filtered_on_duration.get(k, []) + filtered_on_growth_diff.get(k, []) for k in set(filtered_on_duration) | set(filtered_on_growth_diff)}
    decline_events_final_sorted = {k: sorted(v, key=lambda x: x[0][0]) for k, v in decline_events_final.items()}
    if verbose:
        print(f'Number of channels with decline events detected: {len(decline_events_final_sorted)}')

    save_to_csv('data/decline_events_complete.csv', [(channel, event, end_week) for channel, events in decline_events_final_sorted.items() for event, end_week in events], ["Channel", "Event", "Duration"], verbose=verbose)

    mean_subscribers = df_with_rgr_new.reset_index().groupby('channel')['subs'].mean().reset_index()
    mean_subscribers.columns = ['Channel', 'Mean_Number_of_Subscribers']
    mean_subscribers_dict = mean_subscribers.set_index('Channel')['Mean_Number_of_Subscribers'].to_dict()
    filtered_mean_subscribers_dict = {k: v for k, v in mean_subscribers_dict.items() if v > 1e6}

    decline_events_bb = decline_events_final_sorted.copy()
    for k in list(decline_events_bb.keys()):
        if k in filtered_mean_subscribers_dict:
            decline_events_bb[k].append(filtered_mean_subscribers_dict[k])
        else:
            del decline_events_bb[k]

    channel = 'UC-lHJZR3Gqxm24_Vd_AJ5Yw'
    if channel in decline_events_bb:
        nb_events = len(decline_events_bb.get(channel))
    else:
        nb_events = 0
    if verbose:
        print(f'Number of decline events detected for channel {channel}: {nb_events}')

    data = load_bb_timeseries_processed(verbose=True)
    bb_channels = data.index.get_level_values('channel').unique()

    nb_channels = 0
    for channel in bb_channels:
        if channel in decline_events_bb:
            nb_channels += 1
    
    print(f'Number of channels with decline events detected: {nb_channels}')

    lance_stewart  =  'UC6-NBhOCP8DJqnpZE4TNE-A'

    print("With the past technique, every red area was a decline event.")
    plot_rolling_growth_rate2(lance_stewart, df_with_rgr_final)

    print("Now, we obtain less decline events,  but more meaningful ones.")
    plot_new_detection(df_with_rgr_grouped_final, decline_events_final_sorted, lance_stewart)

    save_to_csv('data/bb_from_declined_events.csv', [(channel, event, end_week, events[-1]) for channel, events in decline_events_bb.items() for event, end_week in events[:-1]], ["Channel", "Event", "Duration", "Mean_Number_of_Subscribers"], verbose=verbose)

if __name__ == '__main__':
    create_datasets(verbose=True)
