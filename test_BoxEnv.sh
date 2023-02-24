# #!/bin/bash
# nohup  python -u  main.py --env_name BoxEnv --mode test --log_time $1 --gpu_device $3 --negative_sampling for_test --exp_setting BoxEnv_$2_test > log_experiment/BoxEnv_$2_test.txt 2>&1 &

# # bash test_BoxEnv.sh 1658842393 JustOptimal 2


# nohup python -u  main.py --env_name BoxEnv --min_max_num_agents 6 6 --mode test --log_time 1662011362 --exp_setting BoxEnv_with89_test > log_experiment/BoxEnv_with89_test.txt 2>&1 &

# nohup python -u  main.py --env_name BoxEnv --min_max_num_agents 10 10 --mode test --log_time 1662011362 --exp_setting BoxEnv_with89_10_agent_test > log_experiment/BoxEnv_with89_10Agent_test.txt 2>&1 &

# # python -u  main.py --env_name BoxEnv --mode test --log_time 1658842393 --gpu_device 3 --negative_sampling for_test --exp_setting BoxEnv_just_optimal_test > log_experiment/BoxEnv_just_optimal_test.txt




# # python -u  main.py --env_name BoxEnv --mode test --log_time 1659709676 --gpu_device 3 --negative_sampling for_test --exp_setting BoxEnv_just_optimal_test > log_experiment/BoxEnv_just_optimal_test.txt



# # w='1.005'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &

# # w='1.01'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &

# # w='1.05'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &

# # w='1.1'; nohup taskset -c 1 python -u main.py --env_name BoxEnv --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &





# # TO RUN ocra
# #  nohup taskset -c 1 python -u main.py --env_name BoxEnv --agent EvaluatorORCA --exp_setting BoxTest1-5_$w --min_max_num_agents 1 5 --mode test --log_time 1662011362  --infer_w $w > log_experiment/inference_box_1_5_$w.txt 2>&1 &


# nohup taskset -c 1 python -u main.py --env_name BoxEnv --agent EvaluatorORCA --exp_setting BoxTest_Agent_1_5_ORCA --min_max_num_agents 1 5 --mode test --log_time 1662011362  > log_experiment/inference_box_1_5_OCRA.txt 2>&1 &


# for j in {6..10..2}
# do
#    nohup taskset -c $((j)) python -u main.py --env_name BoxEnv --agent EvaluatorORCA  --exp_setting BoxTest_Agent_$((j)) --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662011362  > log_experiment/inference_box_Agent_$((j)).txt 2>&1 &
# done

# nohup taskset -c 1 python -u main.py --env_name BoxEnv --agent EvaluatorORCA --exp_setting BoxTest_Agent_6_5_ORCA --min_max_num_agents 1 5 --mode test --log_time 1662011362  > log_experiment/inference_box_1_5_OCRA.txt 2>&1 &


for j in {6..10..2}
do
   nohup taskset -c $((j)) python -u main.py --env_name BoxEnv --agent_identifier False   --exp_setting BoxTest_no_agent_identifier_Agent_$((j)) --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662496103  > log_experiment/inference_box_no_agent_identifier_Agent_$((j)).txt 2>&1 &
done

for j in {6..10..2}
do
   nohup taskset -c $((j/2)) python -u main.py --env_name BoxEnv --temporal_encoding False   --exp_setting BoxTest_no_temporal_encoding_Agent_$((j)) --min_max_num_agents $((j)) $((j)) --mode test --log_time 1662477041  > log_experiment/inference_box_no_temporal_encoding_Agent_$((j)).txt 2>&1 &
done