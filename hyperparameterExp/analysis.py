import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def report_results(data_type):
    # Data analysis
    dfs = []
    for f in os.listdir(f'{SCRIPT_DIR}/Results/{data_type}/'):
        dfs.append(pd.read_pickle(f'{SCRIPT_DIR}/Results/{data_type}/{f}'))
    df = pd.concat(dfs)
    df['result_found'] = df['factual_cf_distance'].isna() == False

    df['parameters_str'] = df['parameters'].apply(lambda x: str(x))

    df_gb_greedy = df[df['cf_strategy'] == 'greedy'].groupby(['parameters_str'])

    analysis_greedy = df_gb_greedy.agg({
        'data_type': 'count',
        'cf_time': 'mean',
        'factual_cf_distance': 'mean',
        'factual_cf_not_optimized_distance': 'mean',
        'result_found': 'mean'}).sort_values(
        by=['result_found', 'factual_cf_distance', 'cf_time'],
        ascending=[False, True, True])

    df_gb_random = df[df['cf_strategy'] == 'random'].groupby(['parameters_str'])

    print(f'Best {data_type} Greedy Parameters: {analysis_greedy.iloc[0].name}')

    analysis_random = df_gb_random.agg({
        'data_type': 'count',
        'cf_time': 'mean',
        'factual_cf_distance': 'mean',
        'factual_cf_not_optimized_distance': 'mean',
        'result_found': 'mean'}).sort_values(
        by=['result_found', 'factual_cf_distance', 'cf_time'],
        ascending=[False, True, True])

    print(f'Best {data_type} Random Parameters: {analysis_random.iloc[0].name}')


report_results('tabular')
report_results('text')
report_results('image')