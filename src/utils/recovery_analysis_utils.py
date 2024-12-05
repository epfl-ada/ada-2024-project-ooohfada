import pandas as pd
import numpy as np
import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize
import pickle
import seaborn as sns
from tqdm import tqdm

GREEN = '#2ca02c'
RED = '#d62728'

def match_declines(declines):
    """
    Match declines in the context of a matched observational study.
    """
    declines = declines.copy()

    declines = _preprocess_for_matching(declines)

    declines['Propensity'] = _compute_propensity_score(declines)

    treatment = declines[declines['Recovered'] == 1]
    control = declines[declines['Recovered'] == 0]

    # Match treatment and control
    print('Matching treatment and control')

    graph = nx.Graph()

    # TODO this PSM is very very slow :
    # 1. use caliper?
    # 2. reduce nb of samples ? is random sampling ok or not at all?

    print('Computing similarities')
    similarities = 1 - np.abs(control['Propensity'].values[:, None].T - treatment['Propensity'].values[:, None])

    print('Computing matches')
    matched_index_indices = scipy.optimize.linear_sum_assignment(similarities, maximize=True)

    matched_indices = [(control.index[j], treatment.index[i]) for i, j in zip(*matched_index_indices)]

    return matched_indices

def _compute_propensity_score(declines):
    regressors = ['Duration', 'Subs_start', 'Views_start'] + [col for col in declines.columns if 'Cat' in col]

    model = sm.Logit(declines['Recovered'].astype(np.float64), declines[regressors].astype(np.float64))
    res = model.fit()

    return res.predict()


def _preprocess_for_matching(declines):
    declines['Duration'] = ( declines['Duration'] -  declines['Duration'].mean())/ declines['Duration'].std()
    declines['Subs_start'] = ( declines['Subs_start'] -  declines['Subs_start'].mean())/ declines['Subs_start'].std()
    declines['Views_start'] = ( declines['Views_start'] -  declines['Views_start'].mean())/ declines['Views_start'].std()
    declines = pd.get_dummies(data = declines, columns = ['Category'], prefix = 'Cat', drop_first = True, )

    declines.rename(columns={col: col.replace(' ', '_').replace('&', '_and_') for col in declines.columns}, inplace=True)

    return declines

def _get_similarity(propensity1, propensity2):
    return 1 - np.abs(propensity1 - propensity2)

def plot_groups_by_categories(df):
    plt.figure(figsize=(13, 4))
    ax = plt.subplot(1, 2, 1)

    # show percentage and count of recovered vs not recovered
    counts = df['Recovered'].value_counts(normalize=False)

    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=[RED, GREEN], legend=False)
    plt.title('Recovery after a decline')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.yticks([100000, 200000, 300000], ['100k', '200k', '300k'])
    plt.xlabel('Managed to recover from the decline')
    plt.ylabel('Count')
    plt.ylim(0, max(counts) * 1.1)

    # add text with the percentage
    for i, count in enumerate(counts):
        plt.text(i, count, count, ha='center', va='bottom')

    ax = plt.subplot(1, 2, 2)

    # bar plot with categories
    counts = df.groupby('Category')['Recovered'].value_counts(normalize=True).unstack().fillna(0) * 100
    # add mean line
    mean =  (1 - df['Recovered'].mean()) * 100
    plt.axhline(mean, color='black', linestyle='--', linewidth=1)
    counts.plot(kind='bar', stacked=True, color=[RED, GREEN], ax=ax, legend=False)
    plt.title('Proportion of successful recoveries by category')
    plt.xlabel('Category')
    plt.ylim(0, 100)
    plt.ylabel('Percentage')
    plt.yticks([0, 20, 40, 60, 80, 100], ['0%', '20%', '40%', '60%', '80%', '100%'])

    # put the mean on the right
    ax_right = plt.gca().twinx()
    ax_right.set_ylim(0, 100)
    ax_right.set_yticks([mean])
    ax_right.set_yticklabels([f'{mean:.2f}%'])
    ax.legend([f'Mean over all declines', 'Not recovered', 'Recovered'], loc='lower center')

    plt.show()

def plot_group_distributions(df):
    plt.figure(figsize=(14, 8))

    ax = plt.subplot(2, 2, 1)

    sns.histplot(data=df, x="Subs_start", hue="Recovered", log_scale=True, element="step", palette=[RED, GREEN], ax=ax)

    plt.title('Distribution of channels by subscribers\nat the start of the decline')
    plt.xlabel('Subscribers at the start of the decline')
    plt.ylabel('Number of channels')

    ax = plt.subplot(2, 2, 2)

    sns.histplot(data=df, x="Views_start", hue="Recovered", log_scale=True, element="step", palette=[RED, GREEN], ax=ax)

    plt.title('Distribution of channels by total number of\nviews at the start of the decline')
    plt.xlabel('Views at the start of the decline')
    plt.ylabel('Number of channels')

    ax = plt.subplot(2, 2, 3)

    sns.histplot(data=df, x="Activity_start", hue="Recovered", log_scale=True, element="step", palette=[RED, GREEN], ax=ax)

    plt.title('Distribution of channels by activity\nat the start of the decline')
    plt.xlabel('Activity at the start of the decline')
    plt.ylabel('Number of channels')

    ax = plt.subplot(2, 2, 4)

    sns.histplot(data=df, x="Delta_videos", hue="Recovered", log_scale=True, element="step", palette=[RED, GREEN], ax=ax)

    plt.title('Distribution of channels by delta videos\nat the start of the decline')
    plt.xlabel('Delta videos at the start of the decline')
    plt.ylabel('Number of channels')

    plt.tight_layout()
    plt.show()

def plot_sampling_rates(df, seed):
    # Sample the data at different sampling rates
    sample_proportions = np.linspace(0.01, 1, 100)
    new_dfs = {}
    for prop in sample_proportions:
        new_dfs[prop] = df.sample(frac=prop, replace=False, random_state=seed)

    # Plot the recovery rates
    recovered_props = [new_dfs[prop]['Recovered'].mean() for prop in sample_proportions]
    unrecovered_props = [1 - prop for prop in recovered_props]
    plt.figure(figsize=(6, 2))
    plt.plot(sample_proportions, recovered_props, label='Recovered', color=GREEN)
    plt.plot(sample_proportions, unrecovered_props, label='Not recovered', color=RED)
    plt.xlabel('Sample proportion')
    plt.ylabel('Proportion of declines')
    plt.legend()
    plt.show()

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

def get_matches(df):
    try:
        with open('data/matches.pkl', 'rb') as f:
            matches = pickle.load(f)
            print("Matches loaded from file.")
    except FileNotFoundError:
        print("File not found, computing the matches...")
        matches = match_declines(df)

    # Save the newly computed matches
    with open('data/matches.pkl', 'wb') as f:
        pickle.dump(matches, f)
        print("Matches saved to file.")
