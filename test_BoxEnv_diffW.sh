#!/bin/bash

# w='1.005'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &

# w='1.01'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &

# w='1.05'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &

# w='1.1'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &


# for j in {6..10..2}
# do
#     w='1.005'; nohup taskset -c $((j/2-3)) python -u main.py --env_name BoxEnv --exp_setting BoxTest_Agent_$((j))_$w --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_Agent_$((j))_$w.txt 2>&1 &
# done


# for j in {6..10..2}
# do
#     w='1.01'; nohup taskset -c $((j)) python -u main.py --env_name BoxEnv --exp_setting BoxTest_Agent_$((j))_$w --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_Agent_$((j))_$w.txt 2>&1 &
# done

for j in {6..8..2}
do
    w='1.05'; nohup taskset -c $((j/2-3)) python -u main.py --env_name BoxEnv --exp_setting BoxTest_Agent_$((j))_$w --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_Agent_$((j))_$w.txt 2>&1 &
done

for j in {6..8..2}
do
    w='1.1'; nohup taskset -c $((j)) python -u main.py --env_name BoxEnv --exp_setting BoxTest_Agent_$((j))_$w --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_Agent_$((j))_$w.txt 2>&1 &
done