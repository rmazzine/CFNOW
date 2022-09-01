import sys

sys.path.append('../')

import pandas as pd
from cfbench.cfbench import BenchmarkCF, analyze_results, TOTAL_FACTUAL

from cfnow.cf_finder import find_tabular
from tabularExp.utils import timeout, TimeoutError

# Get initial and final index if provided
if len(sys.argv) == 3:
    initial_idx = sys.argv[1]
    final_idx = sys.argv[2]
else:
    initial_idx = 0
    final_idx = TOTAL_FACTUAL

# Create Benchmark Generator
benchmark_generator = BenchmarkCF(
    show_progress=True,
    disable_gpu=True,
    initial_idx=int(initial_idx),
    final_idx=int(final_idx)).create_generator()


@timeout(600)
def run_experiment(benchmark_data):
    # Get factual array
    factual_array = benchmark_data['factual_oh']
    # Get Keras TensorFlow model
    model = benchmark_data['model']
    # Numerical continuous features
    num_feats = benchmark_data['num_feats']
    # Categorical features
    cat_feats = benchmark_data['cat_feats']

    # OH columns
    oh_cols = list(benchmark_data['df_oh_test'].columns)[:-1]

    feat_types = {f: 'cat' if str(int(f.split('_')[0])) in cat_feats else 'num' for f in oh_cols}

    try:
        # Create CF
        cf_data = find_tabular(
            pd.Series(factual_array, index=oh_cols),
            model.predict,
            feat_types=feat_types,
            has_ohe=True)

        # Get Evaluator
        evaluator = benchmark_data['cf_evaluator']

        # Evaluate CF
        # Simple
        evaluator(
            cf_out=cf_data.cf_not_optimized.tolist(),
            algorithm_name='cfnow_greedy_simple',
            cf_generation_time=cf_data.time_cf_not_optimized,
            save_results=True)
        # Optimized
        evaluator(
            cf_out=cf_data.cf.tolist(),
            algorithm_name='cfnow_greedy',
            cf_generation_time=cf_data.time_cf,
            save_results=True)
    except TimeoutError:
        print('Timeout')


# The Benchmark loop
for benchmark_data in benchmark_generator:
    run_experiment(benchmark_data)
