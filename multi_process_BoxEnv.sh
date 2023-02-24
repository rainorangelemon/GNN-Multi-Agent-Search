#!/bin/bash
# for i in {0..2999..300}
# do
#   nohup taskset -c $(($i/300+21)) python -u datasetgenerator/DatasetGenMultiAgent.py --data_root /local/scratch/ql295/Data/MultiAgentSearch/dataset --env_name BoxEnv --min_max_num_agents 1 5\
#   --ID_Env_Start $i --ID_Env_End $((i+300)) --num_ProblemGraph_PerEnv 20 --num_Cases_PerProblemGraph 1  > log/BoxEnv"_"$i"_"$((i+300)).txt 2>&1 &
# done



for i in {0..2999..1000}
do
  nohup taskset -c $(($i/1000+4)) python -u datasetgenerator/DatasetGenMultiAgent.py --data_root /media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/ --env_name BoxEnv --min_max_num_agents 1 5\
  --ID_Env_Start $i --ID_Env_End $((i+1000)) --num_ProblemGraph_PerEnv 1 --num_Cases_PerProblemGraph 1  > log/BoxEnv"_"$i"_"$((i+1000)).txt 2>&1 &
done



# echo "This will wait until the dataset generation is done"
# date
# wait
# date
# echo "Start splitting the dataset"

# python datasetgenerator/DatasetGenMultiAgent.py --env_name BoxEnv --split_dataset True --min_max_num_agents 1 5
# python -u datasetgenerator/DatasetGenMultiAgent.py --data_root dataset --env_name BoxEnv --min_max_num_agents 1 5 --ID_Env_Start 0 --ID_Env_End 10 --num_ProblemGraph_PerEnv 20 --num_Cases_PerProblemGraph 1
# python -u datasetgenerator/DatasetGenMultiAgent.py --data_root /local/scratch/ql295/Data/MultiAgentSearch/dataset --env_name BoxEnv --min_max_num_agents 1 5 --ID_Env_Start 10 --ID_Env_End 20 --num_ProblemGraph_PerEnv 20 --num_Cases_PerProblemGraph 1


# python -u datasetgenerator/DatasetGenMultiAgent.py --data_root /local/scratch/ql295/Data/MultiAgentSearch/dataset --env_name BoxEnv --min_max_num_agents 1 5 --ID_Env_Start 10 --ID_Env_End 11 --num_ProblemGraph_PerEnv 20 --num_Cases_PerProblemGraph 1



# echo "This will wait until the dataset have been splitted into train, valid and testset"
# date
# wait
# date
# echo "Start convert the dataset into pt file"


# python dataloader/BatchDataloader.py  --env_name BoxEnv --min_max_num_agents 1 5 --mode all --re_preprocess False --over_write_preprocess False

