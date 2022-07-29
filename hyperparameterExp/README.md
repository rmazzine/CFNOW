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

The partitions allow to run the experiments in parallel in multiple machines.