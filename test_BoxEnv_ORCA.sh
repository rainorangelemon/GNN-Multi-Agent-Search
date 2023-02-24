#!/bin/bash


for j in {6..10..2}
do
   nohup taskset -c $((j/2)) python -u main.py --env_name BoxEnv --agent EvaluatorORCA  --exp_setting BoxTest_Agent_$((j)) --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662011362  > log_experiment/inference_box_Agent_$((j))_ORCA.txt 2>&1 &
done

# nohup taskset -c 1 python -u main.py --env_name BoxEnv --agent EvaluatorORCA --exp_setting BoxTest_Agent_6_5_ORCA --min_max_num_agents 1 5 --mode test --log_time 1662011362  > log_experiment/inference_box_1_5_OCRA.txt 2>&1 &

