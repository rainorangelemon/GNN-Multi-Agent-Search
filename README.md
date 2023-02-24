
## Setup VirtualEnv
To install, run the following:
``` bash
conda create -n venv_MASearch python=3.8
conda install numpy
# conda install pyg -c pyg -c conda-forge  # install torch geometric
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg  # install torch geometric 2.0.4

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

pip install seaborn
pip install wandb
pip install shapely
pip install torchsummary
pip install easydict
```

## TODO: 
add instruction for docker



## Dataset Generation
To generate dataset:
``` 
python datasetgenerator/DatasetGenMultiAgent.py --env_name BoxEnv \
--min_max_num_agents 1 5
```
``` 
python datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv \
--min_max_num_agents 10 20

```


To split the dataset after generating it from expert solution:
``` 
python datasetgenerator/DatasetGenMultiAgent.py --env_name BoxEnv --split_dataset True\
--min_max_num_agents 1 5   
```

```
python datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv --split_dataset True\
--min_max_num_agents 10 20
```



## Dataset Post-processing by dataloader
To post-process dataset:
``` 
python dataloader/BatchDataloader.py  --env_name BoxEnv --min_max_num_agents 1 5

```

``` 
python dataloader/BatchDataloader.py  --env_name MazeEnv --min_max_num_agents 10 20

```

To post-process dataset (re_preprocess):
``` 
python dataloader/BatchDataloader.py  --env_name BoxEnv --min_max_num_agents 1 5 --mode train --re_preprocess
```
``` 
python dataloader/BatchDataloader.py  --env_name MazeEnv --min_max_num_agents 10 20 --mode train --re_preprocess
```


## Training
To run the experiment in framework (MultiAgentSearch):

Train from scratch on BoxEnv
``` 
python main.py --env_name BoxEnv --mode train
``` 
<!-- taskset -p -c 8-15 
nohup  python -u  main.py --env_name BoxEnv --mode train --batch_size 8 --gpu_device 1 --max_epoch 10000 > log_experiment/BoxEnv_Sigmoid.txt 2>&1 & -->


Train from scratch on MazeEnv
``` 
python main.py --env_name MazeEnv --mode train
``` 
<!-- 
nohup  python -u main.py --env_name MazeEnv --mode train --min_max_num_agents 10 20 --batch_size 8 --gpu_device 2 --max_epoch 10000 > log_experiment/MazeEnv_Sigmoid.txt 2>&1 & -->



Train by continuing learning
``` 
python main.py --env_name MazeEnv --mode train --con_train True --log_time 1654091511
``` 

## Test
``` 
python main.py --env_name BoxEnv --mode test --log_time 1654091511
``` 
<!-- nohup  python -u main.py --env_name BoxEnv --mode test --log_time 1654091511 > log_experiment/BoxEnv_Sigmod_test.txt 2>&1 &-->
``` 
python main.py --env_name MazeEnv --mode test --log_time 1658153431
``` 
<!-- nohup  python -u main.py --env_name MazeEnv --mode test --log_time 1657835571 > log_experiment/MazeEnv_Sigmod.txt 2>&1 &-->
