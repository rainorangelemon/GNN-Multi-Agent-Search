#!/bin/bash

for j in {12..26..2}
do
   nohup taskset -c 0 python -u main.py --env_name MazeEnv --agent EvaluatorOCRA  --exp_setting MazeTest$((j))_ORCA --min_max_num_agents $((j)) $((j)) --mode test  > log_experiment/inference_maze_$((j))_orca.txt 2>&1
   wait
done

# nohup taskset -c 1 python -u main.py --env_name MazeEnv --agent EvaluatorORCA --exp_setting MazeTest_2_10_ORCA --min_max_num_agents 2 10 --mode test  > log_experiment/inference_Maze_2_10_OCRA.txt 2>&1 &
