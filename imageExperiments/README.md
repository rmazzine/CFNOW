# CFNOW Image experiments

In this branch, we make a benchmark of CFNOW comparing with other 5 methods: LIME-C, SHAP-C, Occlusion, Blur and SEDC.

## Requirements
You must have:
- Python 3.10
- Ubuntu 18.04

### Installing the dependencies
On this directory:
```bash
pip install -r requirements_exp.txt
```

## Running the experiments
On this directory:
```bash
python run_exp.py
```

## Docker Image
You can also run the experiments using a docker image. To build the image first go to the repository root and run:
```bash
docker build -t cfnow_image .
```

Then, run the experiments:
```bash
docker run -it --gpus all --rm -v $(pwd)/imageExperiments/Results:/cfnow/imageExperiments/Results cfnow_image
```