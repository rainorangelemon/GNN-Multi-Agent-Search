#!/bin/bash
# for i in {0..2999..750}
# do
#   nohup taskset -c $(($i/300+1)) python -u datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv\
#   --ID_Env_Start $i --ID_Env_End $((i+750)) --num_ProblemGraph_PerEnv 1 --num_Cases_PerProblemGraph 1  > log/MazeEnv_2_10_Agents"_"$i"_"$((i+750)).txt 2>&1 &
# done

sleep 3000

# generate test set
for j in {16..28..2}
do
    for i in {0..2999..750}
    do
      nohup taskset -c $(($i/750+1)) python -u datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv\
      --ID_Env_Start $((i+637)) --ID_Env_End $((i+662)) --mode test --min_max_num_agents $((j)) $((j)) --num_ProblemGraph_PerEnv 1 --num_Cases_PerProblemGraph 1  > log/MazeEnv_$((j))_Agents"_"$i"_"$((i+750)).txt 2>&1 &
    done
    date
    wait
done