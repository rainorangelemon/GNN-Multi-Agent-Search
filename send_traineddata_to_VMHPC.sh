#!/bin/bash
ListTargetMachine="128.232.69.16:/net/archive/export/ql295/Data/MultiAgentSearch/experiments/AgentExplorerMultiAgentRegion login-gpu.hpc.cam.ac.uk:/home/ql295/rds/hpc-work/experiments/AgentExplorerMultiAgentRegion"
ListEnvType="BoxEnv MazeEnv"


ArrTargetMachine=($ListTargetMachine)
ArrEnvType=($ListEnvType)

# fullname="USER INPUT"
read -p "Enter Target machine (0 - VM; 1 - HPC):" TargetMachine
# user="USER INPUT"
read -p "Enter Env name (0 - BoxEnv; 1 - MazeEnv): " TargetEnv
read -p "Enter Trained network name: " LogExpName
read -p "Enter Trained network log_time: " LogTime


if [[ 0 == ${TargetMachine} ]]
then
    scp -r /local/scratch/ql295/Data/MultiAgentSearch/experiments/AgentExplorerMultiAgentRegion/$LogExpName  ql295@${ArrTargetMachine[$TargetMachine]}/$LogExpName

    ##todo HOW TO ADD custised time
    ssh -K ql295@128.232.69.16 'nohup  python -u  main.py --env_name BoxEnv --mode test --log_time $LogExpName --gpu_device 0 --negative_sampling for_test --exp_setting BoxEnv_$LogExpName_test > log_experiment/BoxEnv_$LogExpName_test.txt 2>&1 &'
else
    sshpass -p "Graphmapf2020!" scp -r /local/scratch/ql295/Data/MultiAgentSearch/experiments/AgentExplorerMultiAgentRegion/$LogExpName  ql295@${ArrTargetMachine[$TargetMachine]}/$LogExpName

fi


