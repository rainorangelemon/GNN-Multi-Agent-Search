#!/bin/bash


for j in {6..10..2}
do
   w='1.1'; nohup taskset -c $((j/2)) python -u main.py --env_name BoxEnv --agent EvaluatorECBS  --exp_setting BoxTest_Agent_$((j)) --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662011362 --infer_w $w > log_experiment/inference_box_Agent_$((j))_ECBS.txt 2>&1 &
done

# w='1.1'; nohup taskset -c 11 python -u main.py --env_name BoxEnv --agent EvaluatorECBS --exp_setting BoxTest_Agent_1_5_ECBS --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_ECBS.txt 2>&1 &
