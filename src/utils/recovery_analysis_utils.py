import pandas as pd
import numpy as np
import statsmodels.api as sm
import networkx as nx
from tqdm import tqdm

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

    for control_index, control_row in tqdm(control.iterrows(), total=len(control)):
        for treatment_index, treatment_row in treatment.iterrows():
            similarity = _get_similarity(control_row['Propensity'], treatment_row['Propensity'])
            graph.add_weighted_edges_from([(control_index, treatment_index, similarity)])

    matches = nx.max_weight_matching(graph)

    return matches

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