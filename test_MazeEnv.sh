#!/bin/bash
nohup  python -u  main.py --env_name MazeEnv --mode test --log_time $1 --gpu_device $3 --negative_sampling just_optimal  --exp_setting MazeEnv_$2_test > log_experiment/MazeEnv_$2_test.txt 2>&1 &


# nohup  python -u  main.py --env_name MazeEnv --mode test --log_time 1659972447 --gpu_device 0 --negative_sampling just_optimal  --exp_setting MazeEnv_justOptimal_test > log_experiment/MazeEnv_justOptimal_test.txt 2>&1 &
