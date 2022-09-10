# Tabular CF Benchmark

## Requirements
* Ubuntu 18.04
* Python 3.10

## How to run
* Install requirements
```bash
pip install -r requirements_exp.txt
```

* Run benchmark
```bash
python run_exp.py
```

## Results
* Results are saved in `cfbench_results` folder

## Docker
You must go to the root folder of the project and run the following command:
```bash
docker build -t cfnow_cfbench .
```
Then, you can run the benchmark with the following command:
```bash
docker run -it --rm -v $(pwd)/benchmark/cfbench_results:/cfnow/benchmark/cfbench_results cfnow_cfbench
```