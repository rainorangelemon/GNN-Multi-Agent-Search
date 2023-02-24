#!/bin/bash
ListTargetMachine="128.232.69.16:/net/archive/export/ql295/Data/MultiAgentSearch/dataset login-gpu.hpc.cam.ac.uk:/home/ql295/rds/hpc-work/GNN_MultiAgentSearch/dataset/"
ListEnvType="BoxEnv MazeEnv"


ArrTargetMachine=($ListTargetMachine)
ArrEnvType=($ListEnvType)

# fullname="USER INPUT"
read -p "Enter Target machine (0 - VM; 1 - HPC):" TargetMachine
# user="USER INPUT"
read -p "Enter Env name (0 - BoxEnv; 1 - MazeEnv): " TargetEnv
read -p "Enter Source Data Path: " PathSource


if [[ 0 == ${TargetMachine} ]]
then
    scp -r /local/scratch/ql295/Data/MultiAgentSearch/dataset/${ArrEnvType[$TargetEnv]}/$PathSource  ql295@${ArrTargetMachine[$TargetMachine]}/${ArrEnvType[$TargetEnv]}/$PathSource 
else
    sshpass -p "Graphmapf2020!" scp -r /local/scratch/ql295/Data/MultiAgentSearch/dataset/${ArrEnvType[$TargetEnv]}/$PathSource  ql295@${ArrTargetMachine[$TargetMachine]}/${ArrEnvType[$TargetEnv]}/$PathSource 
fi


# scp -r /net/archive/export/ql295/Data/MultiAgentSearch/dataset/BoxEnv/dataset_3000_minNumAgent_1_maxNumAgent_5  ql295@excalibur:/local/scratch/ql295/Data/MultiAgentSearch/dataset/BoxEnv/dataset_3000_minNumAgent_1_maxNumAgent_5


# scp -r /net/archive/export/ql295/Data/MultiAgentSearch/dataset/MazeEnv/dataset_3000_minNumAgent_2_maxNumAgent_10  ql295@excalibur:/local/scratch/ql295/Data/MultiAgentSearch/dataset/MazeEnv/dataset_3000_minNumAgent_2_maxNumAgent_10
