#!/bin/bash

for j in {12..26..6}
do
  for k in {0..2}
  do
    nohup taskset -c $((k+5)) python -u main.py --env_name MazeEnv --agent EvaluatorECBS  --exp_setting MazeTest$((j+2*k))_ECBS --min_max_num_agents $((j+2*k)) $((j+2*k)) --mode test --infer_w '1.1' > log_experiment/inference_maze_$((j+2*k))_ecbs.txt 2>&1 &
  done
  wait
done

