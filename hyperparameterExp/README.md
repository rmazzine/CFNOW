# Hyperparameter Test for CFNOW

These scripts aim to test the hyperparameter of CFNOW.

All data and models are downloaded automatically (if not already downloaded). 

The hyperparameters are defined in the file `hyperparameterExp/experiment_parameters_generator.py` in the variables `TABULAR_EXPERIMENTS`, `IMAGE_EXPERIMENTS` and `TEXT_EXPERIMENTS`.

Since there are many hyperparameters, the experiments can be divided into several parts. Also, since tabular, image and text experiments use different models, this must be specified.

All experiments use TensorFlow Keras NN models. However, the model size of tabula data is lower than the image and text data, which makes it possible to run in CPU machines. While the image and text data are large and need (actually, recommended) GPU machines.

## How to run the experiments
1 - Set the following environment variables
```shell
export DATA_TYPE=tabular # or image or text
export NUM_PARTITIONS=5 # This divides the experiment into n (in this case 5) parts.
export PARTITION_ID=1 # This is the id of the partition.
export NUM_SAMPLE_PARAMETERS=100 # Number of parameters to be randomly sampled (for random and greedy). 
```
2 - Run the following command (with terminal in the directory `hyperparameterExp`)
```shell
python main.py
```

## This experiment uses the following parameters
### tabular
* DATA_TYPE: tabular
* NUM_PARTITIONS: 1
* PARTITION_ID: 1
* NUM_SAMPLE_PARAMETERS: 500 (represent 23.15% of total combinations)

### image
* DATA_TYPE: image
* NUM_PARTITIONS: 1
* PARTITION_ID: 1
* NUM_SAMPLE_PARAMETERS: 250 (represent 23.15% of total combinations)

### text
* DATA_TYPE: text
* NUM_PARTITIONS: 1
* PARTITION_ID: 1
* NUM_SAMPLE_PARAMETERS: 250 (represent 23.15% of total combinations)

The partitions allow to run the experiments in parallel in multiple machines.

## Docker run
To run the experiments in Docker, go to the root directory of the project and run the following command:
```shell
docker build -t hyperparameterExp .
dk run -t --gpus all -e DATA_TYPE=tabular -e NUM_PARTITIONS=1 -e PARTITION_ID=1 -e NUM_SAMPLE_PARAMETERS=1 -v $(pwd)/hyperparameterExp/DockerExpData/:/CFNOW/hyperparameterExp/Results hyperparameterExp
```

## Reproducibility
The experiments reported were generated with the following command:
```shell
docker build -t hyperparameterexp .
docker run -t --gpus all -e DATA_TYPE=tabular -e NUM_PARTITIONS=1 -e PARTITION_ID=1 -e NUM_SAMPLE_PARAMETERS=500 -v $(pwd)/hyperparameterExp/DockerExpData/:/CFNOW/hyperparameterExp/Results hyperparameterexp;
docker run -t --gpus all -e DATA_TYPE=image -e NUM_PARTITIONS=1 -e PARTITION_ID=1 -e NUM_SAMPLE_PARAMETERS=250 -v $(pwd)/hyperparameterExp/DockerExpData/:/CFNOW/hyperparameterExp/Results hyperparameterexp;
docker run -t --gpus all -e DATA_TYPE=text -e NUM_PARTITIONS=1 -e PARTITION_ID=1 -e NUM_SAMPLE_PARAMETERS=250 -v $(pwd)/hyperparameterExp/DockerExpData/:/CFNOW/hyperparameterExp/Results hyperparameterexp
```