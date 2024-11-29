import pandas as pd

def match_declines(declines, timeseries, channels):
    """
    Match declines in the context of a matched observational study.
    """
    declines = declines.copy()
    

    declines['Duration'] = ( declines['Duration'] -  declines['Duration'].mean())/ declines['Duration'].std()
    declines['Subs_start'] = ( declines['Subs_start'] -  declines['Subs_start'].mean())/ declines['Subs_start'].std()
    declines['Views_start'] = ( declines['Views_start'] -  declines['Views_start'].mean())/ declines['Views_start'].std()
    declines = pd.get_dummies(data = declines, columns = ['Category'], prefix = 'Cat_')

    return declines