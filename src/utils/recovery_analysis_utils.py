import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.optimize
import pickle
from tqdm import tqdm
import json

def  get_matches(treatment: str, declines: pd.DataFrame, verbose: bool = False):
    """
    Match declines in the context of a matched observational study.
    """

    declines = declines.copy()

    df_treatment = declines[treatment]

    # Preprocess : create dummy variables for the categories and standardize the data
    declines = preprocess_for_matching(declines, treatment)

    # Compute the propensity score
    declines['Propensity'] = _compute_propensity_score(predictors=declines, treatment_values=df_treatment, verbose=verbose)

    treatment_group = declines[df_treatment]
    control_group = declines[~(df_treatment == True)]

    similarities = 1 - np.abs(control_group['Propensity'].values[:, None].T - treatment_group['Propensity'].values[:, None])

    matched_index_indices = scipy.optimize.linear_sum_assignment(similarities, maximize=True)

    matched_indices = [(control_group.index[j], treatment_group.index[i]) for i, j in zip(*matched_index_indices)]

    return matched_indices

def preprocess_for_matching(declines: pd.DataFrame, treatment: str):
    # These columns should not be used to compute the propensity score
    excluded_columns = [treatment, 'Channel', 'Recovered', 'Start', 'End', 'Duration']

    declines = declines.copy().drop(columns=excluded_columns)

    ZERO_DIVISION = 1e-6

    for col in [col for col in declines.columns if col != 'Category']:
        declines[col] = (declines[col] - declines[col].mean()) / (declines[col].std() + ZERO_DIVISION)

    declines = pd.get_dummies(data = declines, columns = ['Category'], prefix = 'Cat', drop_first = True)
    declines.rename(columns={col: col.replace(' ', '_').replace('&', '_and_') for col in declines.columns}, inplace=True)

    return declines

def _compute_propensity_score(predictors: pd.DataFrame, treatment_values: pd.Series, verbose: bool = False):
    model = sm.Logit(treatment_values.astype(np.float64), predictors.astype(np.float64))
    res = model.fit(disp=verbose)

    if verbose:
        print(res.summary())

    return res.predict()

def str_to_list(s):
    elements = s.strip('[]').split(', ')
    return [int(sub) if sub.isnumeric() else sub for sub in elements if sub]

def get_sampled_declines_with_videos(df, df_videos):
    try:
        declines = pd.read_csv('data/sampled_decline_events_with_videos.csv')
        print("Sampled declines with videos loaded from file.")
    except FileNotFoundError:
        print("File not found, computing the sampled declines with videos...")

        df = df.copy()

        def find_videos_before(row):
            channel_mask = df_videos['channel'] == row['Channel']
            left_mask = df_videos['week'] >= row['Start'] - row['Duration']
            right_mask = df_videos['week'] <= row['Start']
            return df_videos[channel_mask & left_mask & right_mask].index.tolist()

        def find_videos_after(row):
            channel_mask = df_videos['channel'] == row['Channel']
            left_mask = df_videos['week'] >= row['Start']
            right_mask = df_videos['week'] <= row['End'] + row['Duration']
            return df_videos[channel_mask & left_mask & right_mask].index.tolist()
            

        df['Videos_before'] = [[]] * len(df)
        df['Videos_after'] = [[]] * len(df)

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            df.at[idx, 'Videos_before'] = find_videos_before(row)
            df.at[idx, 'Videos_after'] = find_videos_after(row)

        declines = df

    declines['Videos_before'] = declines['Videos_before'].apply(str_to_list)
    declines['Videos_after'] = declines['Videos_after'].apply(str_to_list)

    return declines

def add_video_stats(df, df_videos):
    df = df.copy()

    # Compute the number of videos per week before and after each decline
    df['Videos_per_week_before'] = df.apply(lambda row: len(row['Videos_before']) / row['Duration'], axis=1)
    df['Videos_per_week_after'] = df.apply(lambda row: len(row['Videos_after']) / row['Duration'], axis=1)

    # Compute the mean duration of videos before and after each decline
    df['Mean_duration_before'] = df.apply(lambda row: np.mean(df_videos.loc[row['Videos_before'], 'duration']), axis=1)
    df['Mean_duration_after'] = df.apply(lambda row: np.mean(df_videos.loc[row['Videos_after'], 'duration']), axis=1)

    return df

def perform_logistic_regression(X, y):
    # Make X and y numeric, and add a constant
    ZERO_DIVISION = 1e-6

    X = X.astype(float)
    X = (X - X.mean()) / (X.std() + ZERO_DIVISION)
    y = y.astype(float)
    X = sm.add_constant(X)

    # Fit the logistic regression
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=0)

    return results

def map_topics_to_llm_themes(path_llm_topics, df_topics):
    with open(path_llm_topics, "r") as f:
        topic_to_theme = json.load(f)

    # Convert keys and values to string
    df_topics['Topic_before'] = df_topics['Topic_before'].dropna().astype(int).astype(str)
    df_topics['Topic_after'] = df_topics['Topic_after'].dropna().astype(int).astype(str)

    # Mapping
    df_topics['Topic_before'] = df_topics['Topic_before'].map(topic_to_theme)
    df_topics['Topic_after'] = df_topics['Topic_after'].map(topic_to_theme)

    df_topics.dropna()
    return df_topics

def filter_topic_transitions(df):
    # Group by Topic_before and Topic_after to calculate recovery rates
    df = df.groupby(['Topic_before', 'Topic_after']).agg(
        recovery_rate=('Recovered', 'mean'),
        count=('Recovered', 'size')
    ).reset_index()

    # Filter for meaningful transitions, with more than 30 cases
    df_filtered = df[df['count'] > 30]
    return df_filtered

def add_declines_to_db(df, df_channels, df_data_processed):
    df_all_declines = df

    # If the decline is longer than 4 months without recovery, we consider the YouTuber was not successful in handling it.
    # Our aim is to find strategies that lead to quick recoveries, therefore taking more than 4 months would be considered unsuccessful.
    RECOVERY_THRESHOLD = 4 * 4

    # Add the decline outcome
    df_all_declines['Recovered'] = df_all_declines['Duration'] < RECOVERY_THRESHOLD

    # Split the tuple (decline start, decline end) into two separate columns
    df_all_declines['Event'] = df_all_declines['Event'].apply(lambda s: [int(week_id) for week_id in s[1:-1].split(', ')])
    df_all_declines['Start'] = df_all_declines['Event'].apply(lambda e: e[0])
    df_all_declines['End'] = df_all_declines['Event'].apply(lambda e: e[1])
    df_all_declines.drop('Event', axis=1, inplace=True)

    # Add the channel category
    df_all_declines['Category'] = df_all_declines['Channel'].apply(lambda c: df_channels.loc[c]['category_cc'])

    # Add the channel's subs at the start of the decline
    decline_index = list(zip(df_all_declines['Channel'], df_all_declines['Start']))
    df_all_declines['Subs_start'] = df_data_processed.loc[decline_index, 'subs'].values

    # Add the activity at the start of the decline
    df_all_declines['Activity_start'] = df_data_processed.loc[decline_index, 'activity'].values

    # Add the channel's subs at the start of the decline
    df_all_declines['Views_start'] = df_data_processed.loc[decline_index, 'views'].values

    return df_all_declines

def calculate_difference(df, col_after, col_before, new_col):
    """     
    Calculate the difference between two columns and create a new column.
    """
    df[new_col] = df.apply(lambda row: row[col_after] - row[col_before], axis=1)
    return df

def add_change_columns(df, diff_col, before_col, longer_col, shorter_col, tolerance, threshold = 0.5):
    """
    Add flags for significant increases or decreases with a tolerance threshold.
    """
    df[longer_col] = df.apply(lambda row: (row[diff_col]) / np.max([row[before_col], tolerance]) > threshold, axis=1)
    df[shorter_col] = df.apply(lambda row: (row[diff_col]) / np.max([row[before_col], tolerance]) < -threshold, axis=1)
    return df

def print_stats(df, increase_col, decrease_col, metric_name):
    """
    Print statistics for significant increases and decreases.
    Args:
        df (pd.DataFrame): The input dataframe.
        increase_col (str): Column indicating increases.
        decrease_col (str): Column indicating decreases.
        metric_name (str): A name to describe the metric (e.g., 'video duration').
    """
    print(f"\n{df[increase_col].mean() * 100:.2f}% of the channels increased {metric_name} after the start of the decline.")
    print(f"{df[decrease_col].mean() * 100:.2f}% of the channels decreased {metric_name} after the start of the decline.\n")

def merge_and_report_topic_changes(df, topic_filepath):
    """
    Merge topic change data and report the percentage of channels that changed topics.
    Args:
        df (pd.DataFrame): The input dataframe.
        topic_filepath (str): Filepath for topic change data CSV.
    Returns:
        pd.DataFrame: DataFrame after merging topic change data.
    """
    topic_data = pd.read_csv(topic_filepath)
    topic_data.columns = ['Decline', 'Topic_change', 'Topic_before', 'Topic_after']
    df = pd.merge(df, topic_data, left_index=True, right_on='Decline', how='left')
    print(f"{df['Topic_change'].mean() * 100:.2f}% of the channels changed topic after the start of the decline.")
    return df.drop(columns=['Decline', 'Topic_before', 'Topic_after'])

def calculate_correlation_matrix(df, target_column, columns_to_exclude=None):
    """
    Calculates and displays the correlation matrix, sorted by correlation with a target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The column to sort correlations by.
        columns_to_exclude (list): Columns to exclude from correlation computation.

    Returns:
        pd.Series: Sorted correlations with the target column.
    """
    if columns_to_exclude:
        df = df.drop(columns=columns_to_exclude)
    correlation_matrix = df.corr()
    return correlation_matrix[target_column].sort_values(ascending=False)

def build_reaction_dataframe(df_sampled, df_videos_per_week, df_video_duration):
    """
    Build a dataframe for reaction analysis by combining relevant columns and transformations.

    Parameters:
        df_sampled (pd.DataFrame): The main dataframe containing sampled data.
        df_videos_per_week (pd.DataFrame): Dataframe with weekly video information.
        df_video_duration (pd.DataFrame): Dataframe with video duration information.

    Returns:
        pd.DataFrame: A dataframe structured for reaction analysis.
    """
    # Define the columns to keep from df_sampled
    kept_cols = [
        'Channel', 'Duration', 'Start', 'End', 
        'Posted_more', 'Posted_less', 
        'Posted_longer_videos', 'Posted_shorter_videos', 
        'Recovered', 'Mean_duration_difference', 
        'Mean_frequency_difference'
    ]
    
    df_reactions = pd.concat([df_sampled[kept_cols], df_videos_per_week, df_video_duration], axis=1)

    # Transform frequency-based reactions
    df_reactions = add_reaction_column(
        df_reactions, 
        ['Posted_more', 'Posted_less'], 
        'Frequency_reaction'
    )
    
    # Transform video duration-based reactions
    df_reactions = add_reaction_column(
        df_reactions, 
        ['Posted_longer_videos', 'Posted_shorter_videos'], 
        'Video_duration_reaction'
    )

    return df_reactions

def add_reaction_column(df, columns, new_column_name):
    """
    Add a reaction column based on a set of binary variables.

    Parameters:
        df (pd.DataFrame): The dataframe to modify.
        columns (list of str): List of binary column names to combine.
        new_column_name (str): Name of the new reaction column to create.

    Returns:
        pd.DataFrame: Updated dataframe with the new reaction column.
    """
    # Create a 'No_change' column as the inverse of all specified columns
    df['No_change'] = ~df[columns[0]] & ~df[columns[1]]
    
    # Create the reaction column using the dummies
    df[new_column_name] = pd.from_dummies(df[columns + ['No_change']])
    
    # Drop intermediate columns
    df = df.drop(columns + ['No_change'], axis=1)
    
    return df

def match_upload_frequency(df_sampled, df_videos_per_week):
    """
    Match declines based on upload frequency and calculate recovery rates for each bin.
    
    Parameters:
        df_sampled (pd.DataFrame): The main dataframe containing sampled data.
        df_videos_per_week (pd.DataFrame): Dataframe with weekly video information.
    """
    
    # Define bins and labels for upload frequency
    bins = [0, 0.5, 1, 2, 3, 4, 5, 10]
    labels = ['<0.5', '0.5-1', '1-2', '2-3', '3-4', '4-5', '>5']

    df = df_sampled.drop(columns=(['Mean_frequency_difference', 'Posted_more', 'Posted_less']))
    matched_dfs = {}

    df['Frequency_bin'] = pd.cut(df_videos_per_week['Videos_per_week_after'], bins=bins, labels=labels)

    plot_df = pd.DataFrame(columns=['Frequency_bin', 'Recovery_rate'])

    for bin_label in labels:
        df['Is_in_bin'] = (df['Frequency_bin'] == bin_label)
        print(f'Processing bin: {bin_label} ({df["Is_in_bin"].sum() / len(df) * 100:.2f}% of declines)')

        df_dropped = df.drop(columns = ['Frequency_bin'])
        
        # Perform PSM for the treatment of interest
        matches = get_matches(treatment='Is_in_bin', declines=df_dropped, verbose=False)

        matched_indices_flat = [index for match in matches for index in match]
        matched_df = df.loc[matched_indices_flat]

        # Calculate recovery rate for "in bin" (True)
        recovery_rate = matched_df.groupby('Is_in_bin')['Recovered'].mean() * 100

        matched_dfs[bin_label] = matched_df
        if True in recovery_rate.index: # Append recovery rate for the current bin
            plot_df.loc[len(plot_df)] = [bin_label, recovery_rate[True]]
        else:
            plot_df.loc[len(plot_df)] = [bin_label, 0]
            
    return plot_df

def match_video_duration(df_sampled, df_video_duration):
        # Define bins and labels for upload frequency
    duration_bins = [0*60, 5*60, 10*60, 15*60, 20*60, 30*60, 60*60, 120*60]
    duration_labels = ['<5', '5-10', '10-15', '15-20', '20-30', '30-60', '>60']

    df = df_sampled.copy()
    df = df.dropna()

    matched_dfs = {}

    # Bin the video durations
    df['Duration_bin'] = pd.cut(df_video_duration['Mean_duration_after'], bins=duration_bins, labels=duration_labels)
    print(df['Duration_bin'])
    plot_df = pd.DataFrame(columns=['Duration_bin', 'Recovery_rate'])

    bin_counts = df['Duration_bin'].value_counts()
    print("Number of samples in each bin:")
    print(bin_counts)

    for bin_label in duration_labels:
        df['Is_in_bin_duration'] = (df['Duration_bin'] == bin_label)
        df = df.dropna()
        print(f'Processing bin: {bin_label}')

        df_dropped = df.drop(columns = ['Duration_bin']).dropna()
        

        # Scale the data if necessary (especially for numerical columns)
        if (df_dropped == np.inf).any().any() or (df_dropped.isna()).any().any():
            print(f"Data contains NaN or Infinite values for {bin_label}.")
            continue  # Skip this bin if data is problematic

        df_dropped = df_dropped.loc[:, df_dropped.nunique() > 1]


        # Perform PSM for the treatment of interest
        matches = get_matches(treatment='Is_in_bin_duration', declines=df_dropped, verbose=False)

        matched_indices_flat = [index for match in matches for index in match]
        matched_df = df.loc[matched_indices_flat]

        # Calculate recovery rate for "in bin" (True)
        recovery_rate = matched_df.groupby('Is_in_bin_duration')['Recovered'].mean() * 100

        matched_dfs[bin_label] = matched_df
        if True in recovery_rate.index: # Append recovery rate for the current bin
            plot_df.loc[len(plot_df)] = [bin_label, recovery_rate[True]]
        else:
            plot_df.loc[len(plot_df)] = [bin_label, 0]
            
    return plot_df