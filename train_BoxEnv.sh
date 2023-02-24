#!/bin/bash
ListNegativeSampling="around_optimal just_optimal"
ListTrueFalse="False True"

ArrNegativeSampling=($ListNegativeSampling)
ArrTrueFalse=($ListTrueFalse)

# fullname="USER INPUT"
read -p "Enter experiment name: " expSetting
# user="USER INPUT"
read -p "Enter negative sampling (0 - around_optimal; 1 - just_optimal): " IdNegativeSampling
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv
read -p "Enter GPU ID: " IdGPU
read -p "Enter use continue training (1 - True / 0 - False): " conTrain 

if [[ "False" == ${ArrTrueFalse[$conTrain]} ]]
then
    nohup python -u main.py --env_name BoxEnv --mode train --batch_size 4 --max_epoch 10000 --negative_sampling ${ArrNegativeSampling[$IdNegativeSampling]} \
    --exp_setting BoxEnv_${expSetting} --gpu_device $IdGPU > log_experiment/BoxEnv_$expSetting.txt 2>&1 &
    
else
    read -p "Enter use log time: " logtime 
    nohup python -u  main.py --env_name BoxEnv --mode train --batch_size 4 --max_epoch 10000  --negative_sampling ${ArrNegativeSampling[$IdNegativeSampling]} --con_train ${ArrTrueFalse[$conTrain]} \
    --log_time_trained ${logtime} --exp_setting BoxEnv_${expSetting}_conTrain_load_from_${logtime} --gpu_device $IdGPU  > log_experiment/BoxEnv_${expSetting}_conTrain.txt 2>&1 &
fi


# python -u main.py --env_name BoxEnv --mode train --batch_size 6 --max_epoch 10000 --negative_sampling just_optimal --gpu_device 0

# python -u main.py --env_name BoxEnv --mode train --batch_size 6 --max_epoch 10000 --negative_sampling just_optimal --gpu_device 0


# python -u main.py --env_name BoxEnv --mode train --batch_size 6 --max_epoch 10000 --con_train True --negative_sampling just_optimal --log_time_trained 1659709676 --gpu_device 0 --exp_setting BoxEnv_JustOptimal_conTrain_load_from_1659709676



# nohup python -u main.py --env_name BoxEnv --mode train --batch_size 12 --max_epoch 10000 --negative_sampling just_optimal --exp_setting BoxEnv_NewModel --gpu_device 1 > log_experiment/BoxEnv_NewModel.txt 2>&1 &


# nohup python -u main.py --env_name MazeEnv --mode train --batch_size 48 --max_epoch 10000 --min_max_num_agents 2 10 --negative_sampling just_optimal --exp_setting MazeEnv_NewModel --gpu_device 0 > log_experiment/MazeEnv_NewModel.txt 2>&1 &
