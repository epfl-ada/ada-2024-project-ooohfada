import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px

GREEN = '#2ca02c'
RED = '#d62728'

def plot_recovered_by_categories(df, filename=None):
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

    if filename:
        # remove index name
        counts.index.name = None
        counts.to_csv('plot_data/' + filename, index=True)

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

def plot_heatmap_topics(df):
    pivot_data = df.pivot(
        index='Topic_before', 
        columns='Topic_after', 
        values='recovery_rate'
    )

    # Heatmap of the recovery rates by topic transitions
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Recovery Rate'})
    plt.title('Recovery Rate by Topic Transition')
    plt.xlabel('Topic After')
    plt.ylabel('Topic Before')
    plt.show()

def plot_horizontal_barplot(var, df, x, y):
    # Create the bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df,
        x=x,
        y=y,
        palette='viridis'
    )
    plt.title(f'Recovery Rate by {var}')
    plt.xlabel('Recovery Rate')
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()

def plot_barplot_topics_plotly(df):
    df = df.sort_values(by='recovery_rate', ascending=False)

    # Normalize recovery rates for color mapping (range 0-1)
    min_rate = df['recovery_rate'].min()
    max_rate = df['recovery_rate'].max()
    df['normalized_rate'] = (df['recovery_rate'] - min_rate) / (max_rate - min_rate)


    # Invert the normalized rate (so that high recovery rates correspond to cooler colors)
    df['inverted_rate'] = 1 - df['normalized_rate']  # Inversion

    # Generate color gradient using the coolwarm colormap
    def rate_to_color(rate):
        from matplotlib import cm
        from matplotlib.colors import Normalize, to_hex
        cmap = cm.get_cmap('coolwarm')  # Coolwarm colormap (blue to red)
        norm = Normalize(vmin=0, vmax=1)  # Normalize based on 0-1 range
        return to_hex(cmap(norm(rate)))  # Use inverted normalized rate for color mapping

    # Map recovery rates to colors
    min_rate = df['inverted_rate'].min()
    max_rate = df['inverted_rate'].max()
    # Apply the inverted color mapping to the link color
    df['color'] = df['inverted_rate'].apply(rate_to_color)


    # Prepare y-axis labels (row-wise operation)
    df['transition_label'] = df.apply(
        lambda row: f"{row['Topic_before']} -> {row['Topic_after']} (n={row['count']})", axis=1
    )

    # Create the bar chart with Plotly Express
    fig = px.bar(
        df,
        x='recovery_rate',
        y='transition_label',  # Use the prepared labels
        orientation='h',
        title='Recovery Rate by Topic Transition',
        labels={'x': 'Recovery Rate', 'y': 'Topic Transition'},
        hover_data=['count'],
        color=df['color'],  # Assign custom colors
        color_discrete_map="identity"  # Use colors directly
    )

    # Adjust layout
    fig.update_layout(
        height=1000,
        showlegend=False  # Disable legend since each bar has a unique color
    )

    fig.show()

def sankey_diagram(df):
    # Normalize recovery rates for color mapping (range 0-1)
    min_rate = df['recovery_rate'].min()
    max_rate = df['recovery_rate'].max()
    df['normalized_rate'] = (df['recovery_rate'] - min_rate) / (max_rate - min_rate)

    # Invert the normalized rate (so that high recovery rates correspond to cooler colors)
    df['inverted_rate'] = 1 - df['normalized_rate']  # Inversion

    # Generate color gradient using the coolwarm colormap
    def rate_to_color(rate):
        from matplotlib import cm
        from matplotlib.colors import Normalize, to_hex
        cmap = cm.get_cmap('coolwarm')  # Coolwarm colormap (blue to red)
        norm = Normalize(vmin=0, vmax=1)  # Normalize based on 0-1 range
        return to_hex(cmap(norm(rate)))  # Use inverted normalized rate for color mapping

    # Apply the inverted color mapping to the link color
    df['link_color'] = df['inverted_rate'].apply(rate_to_color)

    # Data preparation for Sankey diagram
    df['Topic_before_Label'] = df['Topic_before'] + " (Before)"
    df['Topic_after_Label'] = df['Topic_after'] + " (After)"

    nodes_before = df['Topic_before_Label'].unique()
    nodes_after = df['Topic_after_Label'].unique()
    nodes = list(nodes_before) + list(nodes_after)

    node_indices = {node: idx for idx, node in enumerate(nodes)}

    # Map the topics to their respective indices for source and target
    source = df['Topic_before_Label'].map(node_indices).tolist()
    target = df['Topic_after_Label'].map(node_indices).tolist()
    value = df['count'].tolist()
    link_colors = df['link_color'].tolist()

    # Prepare hover text with relevant info
    hover_texts = [
        f"Transition: {row['Topic_before']} → {row['Topic_after']}<br>"
        f"Recovery Rate: {row['recovery_rate']:.2%}<br>"
        f"Count: {row['count']}"
        for _, row in df.iterrows()
    ]

    # Create Sankey diagram with Plotly
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=10,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="#004AAD" 
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>',
            color=link_colors  # Dynamic link colors based on inverted recovery rates
        )
    )])

    fig.update_traces(
        hoverinfo='all',  # Display all hover information (source, target, value)
    )


    # Callback to highlight outgoing flows dynamically on hover
    fig.update_layout(
        hovermode='closest',  # Ensure closest hover behavior
        title_text="Topic Transitions and Recovery Rates",
        font_size=12,
        height=800,
    )
    # Display the plot
    fig.show()

def scatterplot_reaction(x, y, data, reaction):
    sns.scatterplot(x=x, y=y, data=data)
    plt.title(f'Upload {reaction} vs. Recovery')
    plt.xlabel(f'Upload {reaction} (videos per week)')
    plt.ylabel('Recovery')
    plt.show()

def kdeplot_reaction(df_true, df_false, column, reaction, xlim):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df_true[column], label='Recovered', fill=True, alpha=0.5)
    sns.kdeplot(df_false[column], label='Not Recovered', fill=True, alpha=0.5)
    plt.xlim(-xlim, xlim)
    plt.title(f'Distribution of Video {reaction} by Recovery Status')
    plt.xlabel('Upload {reaction} (videos per week)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def barplot_reaction(y_lim_min, y_lim_max, data, reaction, x, y):
    sns.barplot(data=data, x=x, y=y, errorbar=None)
    plt.title(f'Recovery by {reaction}')
    plt.xlabel(reaction)
    plt.ylabel('Recovery Rate')
    plt.ylim(y_lim_min, y_lim_max)
    plt.show()
