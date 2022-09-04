# This code runs batches of 100 instances while the CPU and memory usage is below a threshold (90%, 80%)
import os
import psutil
from subprocess import Popen

from cfbench.cfbench import TOTAL_FACTUAL

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# Function that measures the current CPU usage
def measure_cpu_usage():
    cpu_usage = psutil.cpu_percent(60)
    print(f'CPU usage: {cpu_usage}%')
    return cpu_usage

# Function that measures the current memory usage
def measure_memory_usage():
    memory_usage = psutil.virtual_memory().percent
    print(f'Memory usage: {memory_usage}%')
    return memory_usage


for cf_strategy in ['greedy', 'random']:
    total_cf_exp_done = 0
    while total_cf_exp_done < TOTAL_FACTUAL:
        # Get current CPU usage
        cpu_usage = measure_cpu_usage()
        # Get current memory usage
        memory_usage = measure_memory_usage()
        print(f'Running {total_cf_exp_done}/{TOTAL_FACTUAL} experiments')

        # If CPU and memory usage is below 90% (CPU) and 80% (memory)
        if cpu_usage < 90 and memory_usage < 80:
            # Run 100 instances
            initial_idx = total_cf_exp_done
            final_idx = total_cf_exp_done + 100
            final_idx = TOTAL_FACTUAL if final_idx > TOTAL_FACTUAL else final_idx

            print('Starting new exp')
            print(f'python {SCRIPT_PATH}/exp.py {cf_strategy} {initial_idx} {final_idx}')

            Popen(f'python {SCRIPT_PATH}/exp.py {cf_strategy} {initial_idx} {final_idx}', shell=True)

            total_cf_exp_done += 100
