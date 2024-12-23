import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import datetime
from IPython.display import clear_output
from tqdm import tqdm
import pandas as pd
from PIL import Image
import io

from src.data.dataloader_functions import load_processed_data

# I. Load the data

def week_index_to_date(week_index):
    """
    Convert a week index to a date, based on the base date of 2015-01-05.

    Parameters:
    week_index (int): The week index to convert to a date.

    Returns:
    datetime.date: The date corresponding to the week index.
    """
    # Base date corresponding to week index 0
    base_date = datetime.datetime(2015, 1, 5)
    
    # Calculate the date by adding the number of weeks to the base date
    target_date = base_date + datetime.timedelta(weeks=int(week_index))
    
    return target_date

# II. Data Analysis On BB Data

# A. Subscribers Analysis

def plot_subs_by_channel(channel, df):
    """
    Plot the subscriber trends for a given channel.

    Parameters:
    channel (str): The channel to plot.
    df (pd.DataFrame): The DataFrame containing the subscriber data.
    """

    plt.figure(figsize=(15, 5))
    
    df = df.xs(channel, level='channel')
    
    sns.lineplot(data=df, x='week', y='subs', label='Subscribers')
    
    plt.xlabel("Week index")
    plt.ylabel("Subscribers")
    plt.title(f"Subscriber Trends for {channel}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def plot_rolling_growth_rate(channel, df, bad_buzz_df, event_name = "Decline"): 
    """
    Plot the rolling growth rate for a given channel, with vertical lines for bad buzz weeks.

    Parameters:
    channel (str): The channel to plot.
    df (pd.DataFrame): The DataFrame containing the subscriber data.
    bad_buzz_df (pd.DataFrame): The DataFrame containing the bad buzz events.
    event_name (str): The name of the event to display on the plot.
    """
    df_plot = df.xs(channel, level='channel')

    clear_output(wait=True)
    plt.figure(figsize=(15, 5))
    
    sns.lineplot(data=df_plot, x='week', y='delta_subs', label='Delta Subscribers', color='blue')
    sns.lineplot(data=df_plot, x='week', y='rolling_growth_rate', label='Rolling Growth Rate', color='orange')
    
    plt.fill_between(df_plot.reset_index()['week'], df_plot['delta_subs'], df_plot['rolling_growth_rate'], 
                    where=(df_plot['delta_subs'] < df_plot['rolling_growth_rate']), 
                    color='red', alpha=0.3, label=f'Potential {event_name}')

    # Plot vertical lines for bad buzz weeks
    bad_buzz_weeks = bad_buzz_df[bad_buzz_df['channel'] == channel]['week']
    for week in bad_buzz_weeks:
        plt.axvline(x=week, color='green', linestyle='--', alpha=0.7, label=event_name if week == bad_buzz_weeks.iloc[0] else "")
    
    plt.xlabel('Week')
    plt.ylabel('Growth Rate')
    plt.title(f'Delta Subscribers and Rolling Growth Rate for Channel {channel}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def plot_rolling_growth_rate2(channel, df, event_name = "Decline"): 
    """
    Plot the rolling growth rate for a given channel, with vertical lines for decline events weeks.

    Parameters:
    channel (str): The channel to plot.
    df (pd.DataFrame): The DataFrame containing the subscriber data.
    bad_buzz_df (pd.DataFrame): The DataFrame containing the bad buzz events.
    event_name (str): The name of the event to display on the plot.
    """

    df_plot = df.xs(channel, level='channel')

    clear_output(wait=True)
    plt.figure(figsize=(15, 5))
    
    sns.lineplot(data=df_plot, x='week', y='delta_subs', label='Delta Subscribers', color='blue')
    sns.lineplot(data=df_plot, x='week', y='rolling_growth_rate', label='Rolling Growth Rate', color='orange')
    
    plt.fill_between(df_plot.reset_index()['week'], df_plot['delta_subs'], df_plot['rolling_growth_rate'], 
                    where=(df_plot['delta_subs'] < df_plot['rolling_growth_rate']), 
                    color='red', alpha=0.3, label=f'Potential {event_name}')
    
    plt.xlabel('Week')
    plt.ylabel('Growth Rate')
    plt.title(f'Delta Subscribers and Rolling Growth Rate for Channel {channel}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# B. Views Analysis around Detected Bad Buzz 

# 1. Delta Views Evolution

def plot_views_around_bad_buzz(channel, df, bad_buzz_df): 
    """
    Plot the delta views around bad buzz weeks for a given channel.

    Parameters:
    channel (str): The channel to plot.
    df (pd.DataFrame): The DataFrame containing the views data.
    bad_buzz_df (pd.DataFrame): The DataFrame containing the bad buzz events.
    """

    df_plot = df.xs(channel, level='channel').reset_index()
    
    clear_output(wait=True)
    plt.figure(figsize=(15, 5))
    
    sns.lineplot(data=df_plot, x='week', y='delta_views', label='Delta Views', color='blue')
    
    # Plot vertical lines for bad buzz weeks
    bad_buzz_weeks = bad_buzz_df[bad_buzz_df['channel'] == channel]['week']
    for week in bad_buzz_weeks:
        plt.axvline(x=week, color='green', linestyle='--', alpha=0.7, label='Bad Buzz' if week == bad_buzz_weeks.iloc[0] else "")
    
    plt.xlabel('Week')
    plt.ylabel('Views')
    plt.title(f'Views Around Bad Buzz Weeks for Channel {channel}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# 2. One Channel

def normalize_week_indices(df, week):
    """
    Normalize the week indices relative to a given week.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the views data.
    week (int): The week index to normalize the data around.

    Returns:
    pd.DataFrame: The DataFrame with the 'relative_week' column added.
    """

    df['relative_week'] = df['week'] - week

    return df

def define_periods(df):
    """
    Define the pre-buzz, during-buzz, and post-buzz periods based on the relative week index.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the views data.

    Returns:
    pd.DataFrame: DataFrames for the pre-buzz, during-buzz, and post-buzz periods.
    """

    pre_buzz = df[(df['relative_week'] < 0) & (df['relative_week'] >= -10)]
    during_buzz = df[(df['relative_week'] >= 0) & (df['relative_week'] <= 10)]
    post_buzz = df[(df['relative_week'] > 10) & (df['relative_week'] <= 20)]

    return pre_buzz, during_buzz, post_buzz

def calculate_averages(pre_buzz, during_buzz, post_buzz):
    """
    Calculate the average delta views for the pre-buzz, during-buzz, and post-buzz periods.

    Parameters:
    pre_buzz (pd.DataFrame): DataFrame for the pre-buzz period.
    during_buzz (pd.DataFrame): DataFrame for the during-buzz period.
    post_buzz (pd.DataFrame): DataFrame for the post-buzz period.

    Returns:
    tuple: Tuples containing the average delta views for the pre-buzz, during-buzz, and post-buzz periods.
    """

    avg_pre_buzz_views = pre_buzz['delta_views'].mean()
    avg_during_buzz_views = during_buzz['delta_views'].mean()
    avg_post_buzz_views = post_buzz['delta_views'].mean()

    return avg_pre_buzz_views, avg_during_buzz_views, avg_post_buzz_views

def perform_t_test(pre_buzz, during_buzz):
    """
    Perform a t-test to compare the delta views between the pre-buzz and during-buzz periods.

    Parameters:
    pre_buzz (pd.DataFrame): DataFrame for the pre-buzz period.
    during_buzz (pd.DataFrame): DataFrame for the during-buzz period.

    Returns:
    float: The p-value from the t-test.
    """

    t_stat, p_value = ttest_ind(during_buzz['delta_views'], pre_buzz['delta_views'], equal_var=False)

    return p_value

def plot_results(ax, df, avg_pre_buzz_views, week, p_value, rolling_window=10, pos_p_value=-0.15, event_name = "Decline"):
    """
    Plot the delta views around a bad buzz event, displaying the pre-buzz average and t-test results.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to plot on.
    df (pd.DataFrame): The DataFrame containing the views data.
    avg_pre_buzz_views (float): The average delta views for the pre-buzz period.
    week (int): The week index of the bad buzz event.
    p_value (float): The p-value from the t-test.
    rolling_window (int): The window size for the rolling average.
    pos_p_value (float): The vertical position for displaying the p-value.
    event_name (str): The name of the event to display on the plot.
    """

    ax.plot(df['relative_week'], df['delta_views'], label='Delta Views', color='blue')
    ax.plot(df['relative_week'], df['delta_views'].rolling(rolling_window).mean(), label='Rolling Avg (Views)', color='orange')
    ax.axvline(x=0, color='red', linestyle='--', label=f'{event_name} Start')
    ax.axhline(y=avg_pre_buzz_views, color='green', linestyle='--', label=f'Pre-{event_name} Avg')
    
    ax.set_title(f"Delta Views around {event_name} (Week {week})")
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Delta Views')

    ax.text(0.5, pos_p_value, f'p-value: {p_value:.4f}', transform=ax.transAxes, ha='center', va='center')

def analyze_views_around_bad_buzz(channel, df, bad_buzz_df, pos_p_value=-0.25, event_name = "Decline"):
    """
    Plot the delta views around bad buzz weeks for a given channel.

    Parameters:
    channel (str): The channel to plot.
    df (pd.DataFrame): The DataFrame containing the views data.
    bad_buzz_df (pd.DataFrame): The DataFrame containing the bad buzz events.
    pos_p_value (float): The vertical position for displaying the p-value.
    event_name (str): The name of the event to display on the plot.
    """

    df_plot = df.xs(channel, level='channel').reset_index()
    bad_buzz_weeks = bad_buzz_df[bad_buzz_df['channel'] == channel]['week']
    
    fig, axes = plt.subplots(6, 4, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, week in enumerate(bad_buzz_weeks):
        df_plot = normalize_week_indices(df_plot, week)
        pre_buzz, during_buzz, post_buzz = define_periods(df_plot)
        avg_pre_buzz_views, avg_during_buzz_views, avg_post_buzz_views = calculate_averages(pre_buzz, during_buzz, post_buzz)
        
        p_value = perform_t_test(pre_buzz, during_buzz)
        
        plot_results(axes[i], df_plot, avg_pre_buzz_views, week, p_value, pos_p_value=pos_p_value, event_name=event_name)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.75, top=0.95)
    plt.show()

# C. Likes/Dislikes Analysis Around Detected Bad Buzz

def calculate_engagement_averages(pre_buzz, during_buzz):
    """
    Calculate the average likes, dislikes, and like/dislike ratio for the pre-buzz and during-buzz periods.

    Parameters:
    pre_buzz (pd.DataFrame): DataFrame for the pre-buzz period.
    during_buzz (pd.DataFrame): DataFrame for the during-buzz period.

    Returns:
    tuple: Tuples containing the average likes, dislikes, and like/dislike ratio for the pre-buzz and during-buzz periods.
    """

    avg_pre_likes = pre_buzz['like_count'].mean()
    avg_pre_dislikes = pre_buzz['dislike_count'].mean()
    avg_pre_ratio = (pre_buzz['like_count'] / (pre_buzz['like_count'] + pre_buzz['dislike_count'])).mean()

    avg_during_likes = during_buzz['like_count'].mean()
    avg_during_dislikes = during_buzz['dislike_count'].mean()
    avg_during_ratio = (during_buzz['like_count'] / (during_buzz['like_count'] + during_buzz['dislike_count'])).mean()
    
    return avg_pre_likes, avg_pre_dislikes, avg_pre_ratio, avg_during_likes, avg_during_dislikes, avg_during_ratio

def perform_engagement_t_tests(pre_buzz, during_buzz):
    """
    Perform t-tests to compare the dislikes and like/dislike ratios between the pre-buzz and during-buzz periods.

    Parameters:
    pre_buzz (pd.DataFrame): DataFrame for the pre-buzz period.
    during_buzz (pd.DataFrame): DataFrame for the during-buzz period.
    
    Returns:
    tuple: Tuples containing the p-values for the dislikes and like/dislike ratio comparisons.
    """

    t_stat_dislikes, p_value_dislikes = ttest_ind(during_buzz['dislike_count'], pre_buzz['dislike_count'], equal_var=False)
    t_stat_ratio, p_value_ratio = ttest_ind(
        during_buzz['like_count'] / (during_buzz['like_count'] + during_buzz['dislike_count']),
        pre_buzz['like_count'] / (pre_buzz['like_count'] + pre_buzz['dislike_count']),
        equal_var=False
    )

    return p_value_dislikes, p_value_ratio

def plot_engagement_results(ax, df, week, p_value_dislikes, p_value_ratio, pos_p_value=-0.15):
    """
    Plot the likes, dislikes, and like/dislike ratio around a bad buzz event, displaying the t-test results.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to plot on.
    df (pd.DataFrame): The DataFrame containing the engagement data.
    week (int): The week index of the bad buzz event.
    p_value_dislikes (float): The p-value from the dislikes t-test.
    p_value_ratio (float): The p-value from the like/dislike ratio t-test.
    pos_p_value (float): The vertical position for displaying the p-value.
    """
    sns.lineplot(data=df, x='relative_week', y='like_count', label='Likes', color='blue', ax=ax)
    sns.lineplot(data=df, x='relative_week', y='dislike_count', label='Dislikes', color='orange', ax=ax)
    ax2 = ax.twinx()
    ax2.plot(df['relative_week'], (df['like_count'] / (df['like_count'] + df['dislike_count'])), label='Like/Dislike Ratio', color='purple')
    
    ax.axvline(x=0, color='green', linestyle='--', label='Bad Buzz Start')
    
    ax.set_title(f"Likes, Dislikes, and Ratio around Bad Buzz (Week {week})")
    ax.set_xlabel('Weeks Relative to Bad Buzz')
    ax.set_ylabel('Engagement Counts')
    ax2.set_ylabel('Like/Dislike Ratio')
    
    ax.get_legend().remove()
    
    ax.text(0.5, pos_p_value, f'p-value (dislikes): {p_value_dislikes:.4f}, p-value (ratio): {p_value_ratio:.4f}', transform=ax.transAxes, ha='center', va='center')

def analyze_engagement_around_bad_buzz(channel, df, bad_buzz_df, pos_p_value=-0.25):
    """
    Plot the likes, dislikes, and like/dislike ratio around bad buzz weeks for a given channel.

    Parameters:
    channel (str): The channel to plot.
    df (pd.DataFrame): The DataFrame containing the engagement data.
    bad_buzz_df (pd.DataFrame): The DataFrame containing the bad buzz events.
    pos_p_value (float): The vertical position for displaying the p-value.
    """
    df_plot = df.xs(channel, level='channel').reset_index()

    bad_buzz_weeks = bad_buzz_df[bad_buzz_df['channel'] == channel]['week']

    fig, axes = plt.subplots(6, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, week in enumerate(bad_buzz_weeks):
        df_plot = normalize_week_indices(df_plot, week)
        pre_buzz, during_buzz, post_buzz = define_periods(df_plot)
        
        p_value_dislikes, p_value_ratio = perform_engagement_t_tests(pre_buzz, during_buzz)

        plot_engagement_results(axes[i], df_plot, week, p_value_dislikes, p_value_ratio, pos_p_value)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])    

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines + [plt.Line2D([0], [0], color="purple", lw=1.5)], labels + ["Like/Dislike Ratio"], loc='upper center', ncol=4)
    fig.legend(lines + [plt.Line2D([0], [0], color="purple", lw=1.5)], labels + ["Like/Dislike Ratio"], loc='upper center', ncol=4)

    plt.subplots_adjust(hspace=0.35, top=0.95)
    plt.tight_layout()
    plt.show()

# D. Activity around the BB 

def calculate_activity_averages(pre_buzz, during_buzz, post_buzz):
    """
    Calculate the average number of videos posted for the pre-buzz, during-buzz, and post-buzz periods.

    Parameters:
    pre_buzz (pd.DataFrame): DataFrame for the pre-buzz period.
    during_buzz (pd.DataFrame): DataFrame for the during-buzz period.
    post_buzz (pd.DataFrame): DataFrame for the post-buzz period.

    Returns:
    tuple: Tuples containing the average number of videos posted for the pre-buzz, during-buzz, and post-buzz periods.
    """

    avg_pre_activity = pre_buzz['activity'].mean()
    avg_during_activity = during_buzz['activity'].mean()
    avg_post_activity = post_buzz['activity'].mean()

    return avg_pre_activity, avg_during_activity, avg_post_activity

def perform_activity_t_tests(pre_buzz, during_buzz):
    """
    Perform a t-test to compare the number of videos posted between the pre-buzz and during-buzz periods.

    Parameters:
    pre_buzz (pd.DataFrame): DataFrame for the pre-buzz period.
    during_buzz (pd.DataFrame): DataFrame for the during-buzz period.

    Returns:
    float: The p-value from the t-test.
    """

    t_stat_activity, p_value_activity = ttest_ind(during_buzz['activity'], pre_buzz['activity'], equal_var=False)

    return p_value_activity

def plot_activity_results(ax, df, avg_pre_activity, week, p_value_activity, pos_p_value=-0.15):
    """
    Plot the number of videos posted around a bad buzz event, displaying the pre-buzz average and t-test results.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to plot on.
    df (pd.DataFrame): The DataFrame containing the activity data.
    avg_pre_activity (float): The average number of videos posted for the pre-buzz period.
    week (int): The week index of the bad buzz event.
    p_value_activity (float): The p-value from the t-test.
    pos_p_value (float): The vertical position for displaying the p-value.
    """

    sns.lineplot(data=df, x='relative_week', y='activity', label='Activity', color='blue', ax=ax)
    ax.axvline(x=0, color='green', linestyle='--', label='Bad Buzz Start')
    ax.axhline(y=avg_pre_activity, color='red', linestyle='--', label='Pre-Buzz Avg')
    
    ax.set_title(f"Activity around Bad Buzz (Week {week})")
    ax.set_xlabel('Weeks Relative to Bad Buzz')
    ax.set_ylabel('Number of Videos Posted')
    ax.legend()
    
    ax.text(0.5, pos_p_value, f'p-value (dislikes): {p_value_activity:.4f}', transform=ax.transAxes, ha='center', va='center')

def analyze_activity_around_bad_buzz(channel, df, bad_buzz_df, pos_p_value=-0.25):
    """
    Plot the number of videos posted around bad buzz weeks for a given channel.

    Parameters:
    channel (str): The channel to plot.
    df (pd.DataFrame): The DataFrame containing the activity data.
    bad_buzz_df (pd.DataFrame): The DataFrame containing the bad buzz events.
    pos_p_value (float): The vertical position for displaying the p-value.
    """

    df_plot = df.xs(channel, level='channel').reset_index()
    bad_buzz_weeks = bad_buzz_df[bad_buzz_df['channel'] == channel]['week']
    
    fig, axes = plt.subplots(6, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, week in enumerate(bad_buzz_weeks):
        df_plot = normalize_week_indices(df_plot, week)
        pre_buzz, during_buzz, post_buzz = define_periods(df_plot)
        avg_pre_activity, avg_during_activity, avg_post_activity = calculate_activity_averages(pre_buzz, during_buzz, post_buzz)
        
        p_value_activity = perform_activity_t_tests(pre_buzz, during_buzz)
        
        plot_activity_results(axes[i], df_plot, avg_pre_activity, week, p_value_activity, pos_p_value)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.75)
    plt.show()

# Milestone 3
    
def compute_rolling_growth_rate(window=20, verbose=True):
    """
    Compute the rolling growth rate for each channel in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the subscriber data.
    window (int): The window size for the rolling growth rate calculation.

    Returns:
    pd.DataFrame: The DataFrame with the rolling growth rate values.
    """

    if verbose:
        print(f'Loading processed data...')

    # Load the original data for the index
    original_data = load_processed_data(verbose = True)

    if verbose:
        print('Processed data loaded. \n')

    if verbose:
        print(f'Computing rolling growth rate with window size {window}...')

    ROLLING_WINDOW = window  # Set the rolling window for the growth rate, (5 months by default, could be changed in the future)

    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame()

    # Iterate over each group with a progress bar
    for name, group in tqdm(original_data.groupby('channel'), desc="Processing channels"):
        group['rolling_growth_rate'] = group['delta_subs'].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        result = pd.concat([result, group])

    result['growth_diff'] = result['delta_subs'] - result['rolling_growth_rate']

    if verbose:
        print('Rolling growth rate computed. \n')

    if verbose:
        print('Saving the results...')

    result.to_csv('data/df_with_rgr_new.tsv', sep='\t', index=True)

    if verbose:
        print('Results saved in data/df_with_rgr_new.tsv. \n')

def plot_new_detection(df_with_rgr_grouped_final, decline_events_final_sorted, channel):
    """
    Plot the growth difference and rolling growth rate for a given channel, highlighting decline events. Finally save the plot as an HTML file.
    
    Parameters:
    df_with_rgr_grouped_final (pd.DataFrameGroupBy): The grouped DataFrame containing the rolling growth rate data.
    decline_events_final_sorted (dict): The dictionary containing the sorted decline events.
    channel (str): The channel to plot.
    """
    channel_data = df_with_rgr_grouped_final.get_group(channel)

    fig = go.Figure()

    # Highlight decline events - add these first
    for event in decline_events_final_sorted.get(channel , []):
        if isinstance(event, tuple) and len(event) == 2:
            fig.add_vrect(
                x0=event[0][0], 
                x1=event[0][1], 
                fillcolor='#DFC5FE', 
                opacity=0.5, 
                line_width=0,
                layer='below',
                name='Decline event'
            )
        
            
    fig.add_trace(go.Scatter(
            x=[None],  # Empty data to not affect the chart
            y=[None], 
            mode='lines', 
            line=dict(color='#DFC5FE', width=10), 
            name=f'Decline Event'  # Adjust legend name
    ))

    # Add growth difference trace with a specific blue color - add these after
    fig.add_trace(go.Scatter(
        x=channel_data['week'], 
        y=channel_data['delta_subs'], 
        mode='lines', 
        name='Growth diff',
        line=dict(color='#004AAD')  # Specify the color here
    ))

    # Add rolling growth rate trace with a specific red color
    fig.add_trace(go.Scatter(
        x=channel_data['week'], 
        y=channel_data['rolling_growth_rate'], 
        mode='lines', 
        name='Rolling growth rate',
        line=dict(color='#FF0000')  # Specify the color here
    ))

    # Update axes, layout, and other styling
    fig.update_xaxes(
        ticks='outside', 
        tickvals=np.arange(0, 260, 10), 
        ticktext=[str(i) for i in np.arange(0, 260, 10)]
    )
    fig.update_yaxes(ticks='outside')

    fig.update_layout(
        title=f'Growth diff and rolling growth rate for Lance Stewart\'s channel',
        # center the title
        title_x=0.5,
        xaxis_title='Week',
        yaxis_title='Subscribers',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='grey',
            borderwidth=1.5
        ),
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_zeroline=False,
        margin=dict(l=50, r=50, t=100, b=50),
        xaxis_tickcolor='black',
        yaxis_tickcolor='black',
        autosize=False,
        width=800,
        height=600,
    )

    # Add a rectangle shape to create the border
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        xref='paper', yref='paper',
        line=dict(color="grey", width=2)
    )

    # Save the Plotly plot as an HTML file
    pio.write_html(fig, file="plot_data/plot_lancet.html", auto_open=False)

    # Convert and display the corresponding static plot (not interactive)
    image_bytes = fig.to_image(format="png", width=1200, height=800, scale=1)

    # Display the image
    image = Image.open(io.BytesIO(image_bytes))

    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

