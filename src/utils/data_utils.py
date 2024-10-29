def detect_bad_buzzes(df, threshold=-0.1):
    # Sort by channel and datetime to ensure chronological order
    df = df.sort_values(by=['channel', 'datetime']).reset_index(drop=True)
    
    # Calculate the percentage change in subscribers for each channel over time
    df['subs_pct_change'] = df.groupby('channel')['subs'].pct_change()

    # Filter rows where the percentage change is below the threshold
    bad_buzzes = df[(df['subs_pct_change'] < threshold) & (df['subs_pct_change'].notna())]
    
    # Select relevant columns and group by channel
    result = bad_buzzes[['channel', 'datetime', 'subs_pct_change']].groupby('channel').agg(list)
    
    return result

# TODO update and use this file