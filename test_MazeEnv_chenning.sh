#!/bin/bash
# test set
for j in {12..26..2}
do
    counter=1
    for w in {1.005,1.01,1.05,1.1}
    do
        nohup taskset -c $((counter)) python -u main.py --infer_w $w --env_name MazeEnv --exp_setting MazeTest$((j))_$w --min_max_num_agents $((j)) $((j)) --mode test --log_time 1661901774  > log_experiment/inference_maze_$((j))_w_$w.txt 2>&1 &
        $((counter++))
    done
    wait
done