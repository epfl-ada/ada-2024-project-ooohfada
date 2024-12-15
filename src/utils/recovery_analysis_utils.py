import pandas as pd
import numpy as np
import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.optimize
import pickle
import plotly.graph_objects as go
import seaborn as sns
from tqdm import tqdm

GREEN = '#2ca02c'
RED = '#d62728'

def  _match_on(treatment: str, declines: pd.DataFrame, verbose: bool = False):
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

    print('Computing similarities')
    similarities = 1 - np.abs(control_group['Propensity'].values[:, None].T - treatment_group['Propensity'].values[:, None])

    print('Computing matches')
    matched_index_indices = scipy.optimize.linear_sum_assignment(similarities, maximize=True)

    matched_indices = [(control_group.index[j], treatment_group.index[i]) for i, j in zip(*matched_index_indices)]

    return matched_indices

def preprocess_for_matching(declines: pd.DataFrame, treatment: str):
    # These columns should not be used to compute the propensity score
    excluded_columns = [treatment, 'Channel', 'Recovered', 'Start', 'End', 'Duration']

    declines = declines.copy().drop(columns=excluded_columns)

    for col in [col for col in declines.columns if col != 'Category']:
        declines[col] = (declines[col] - declines[col].mean()) / declines[col].std()

    declines = pd.get_dummies(data = declines, columns = ['Category'], prefix = 'Cat', drop_first = True)
    declines.rename(columns={col: col.replace(' ', '_').replace('&', '_and_') for col in declines.columns}, inplace=True)

    return declines

def _compute_propensity_score(predictors: pd.DataFrame, treatment_values: pd.Series, verbose: bool = False):
    model = sm.Logit(treatment_values.astype(np.float64), predictors.astype(np.float64))
    res = model.fit(disp=verbose)

    if verbose:
        print(res.summary())

    return res.predict()

def plot_recovered_by_categories(df):
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
    plt.figure(figsize=(13, 6))

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

    sns.histplot(data=df, x="Duration", hue="Recovered", log_scale=True, element="step", palette=[RED, GREEN], ax=ax)

    plt.title('Distribution of declines by duration')
    plt.xlabel('Duration of the decline')
    plt.ylabel('Number of declines')

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
    plt.ylabel('Proportion')
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

def get_matches(treatment: str, declines: pd.DataFrame, verbose: bool = False):
    try:
        with open(f'data/matches/{treatment}.pkl', 'rb') as f:
            matches = pickle.load(f)
            print(f"Matches loaded from file for treatment {treatment}.")
    except FileNotFoundError:
        print(f"File not found, computing the matches for treatment {treatment}...")
        matches = _match_on(treatment, declines, verbose=verbose)

        # Save the newly computed matches
        with open(f'data/matches/{treatment}.pkl', 'wb') as f:
            pickle.dump(matches, f)
            print("Matches saved to file.")

    return matches

def plot_treatment_effect(df, treatment: str, ax=None):
    if ax is None:
        plt.figure(figsize=(4, 3))

    ax = ax if ax is not None else plt.gca()

    # bar plot with categories
    counts = df.groupby(treatment)['Recovered'].mean() * 100
    # add mean line
    counts.plot(kind='bar', color=[RED, GREEN], legend=False, ax=ax)
    ax.set_title(f'Proportion of successful recoveries\ndepending on {treatment}')
    ax.set_xlabel(treatment)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Recovery rate')
    for i, count in enumerate(counts):
        ax.text(i, count, f'{count:.2f}%', ha='center', va='bottom')
    ax.set_yticks([0, 20, 40, 60, 80, 100], ['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.tick_params(axis='x', rotation=0)

def perform_logistic_regression(X, y):
    # Make X and y numeric, and add a constant
    X = X.astype(float)
    y = y.astype(float)
    X = sm.add_constant(X)

    # Fit the logistic regression
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=0)

    return results

def plot_logit_coefficients(logit_result, title=None, ax=None, color_legend=True, filename=None):

    if ax is None:
        ax = plt.gca()

    # use p-values as the palette
    cmap = plt.cm.coolwarm
    reg_data = pd.DataFrame({'coeff': logit_result.params, 'p-value': logit_result.pvalues, 'se': logit_result.bse.values}).sort_values('coeff', ascending=True)
    reg_data.index = reg_data.index.str.replace('_', ' ').str.replace('const', 'Intercept')
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=1)
    colors = cmap(norm(reg_data['p-value']))

    ax.vlines(0, 0, len(reg_data), color='grey', alpha=0.75, linestyle='--', linewidth=0.5)

    ax.barh(reg_data.index, reg_data['coeff'], color=colors, height=0.6)
    ax.set_title(title if title else 'Logistic regression coefficients for recovery')
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Feature')
    ax.grid(axis='y', linestyle='--', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # add the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('p-value')

    if not color_legend:
        cbar.remove()

    if filename:
        plt.savefig('plot_data/' + filename, bbox_inches='tight')

def plot_coeffs_comparison_by_removing_no_videos_declines(results_all_declines, results_without_no_videos_declines):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    fig.suptitle('Logistic regression coefficients for recovery')

    plot_logit_coefficients(results_all_declines, title='All declines', ax=axes[0], color_legend=False)
    plot_logit_coefficients(results_without_no_videos_declines, title='Without declines with no videos', ax=axes[1])

    plt.tight_layout()
    plt.show()

def plot_distribution_by_frequency_reaction(df_post_freq, column, title):
    avg_col_value = np.log(df_post_freq[column].mean())

    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(10, 2))
    [ax.grid(True) for ax in axes]

    colors = plt.cm.coolwarm([1, 0.2, 0])

    for ax, reaction, color in zip(axes, df_post_freq['Frequency_reaction'].unique(), colors):
        mask = df_post_freq['Frequency_reaction'] == reaction
        sns.histplot(x=column, data=df_post_freq[mask], bins=20, log_scale=True, stat='density', common_norm=False, ax=ax, color=color, label='Before')
        ax.axvline(avg_col_value, color='red', linestyle='--', label='Overall average', linewidth=1)
        ax.set_title(f'YouTuber reaction : {reaction.replace("_", " ")}')
        ax.set_xlabel('Videos per week')
        handles, labels = ax.get_legend_handles_labels() 
        handles, labels = [handles[0]], [labels[0]]
        fig.legend(handles, labels, loc='upper right')
        
    fig.suptitle(title)
    plt.tight_layout()