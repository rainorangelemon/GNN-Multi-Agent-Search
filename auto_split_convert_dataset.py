import time
import os

start_time = time.time()
previous_time = 0


while True:
    if (time.time() - previous_time) > 60*30:
        os.system('python datasetgenerator/DatasetGenMultiAgent.py --env_name BoxEnv --split_dataset True --min_max_num_agents 1 5')
        os.system('python dataloader/BatchDataloader.py  --env_name BoxEnv --min_max_num_agents 1 5 --mode all --over_write_preprocess False')
        os.system('python datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv --split_dataset True --min_max_num_agents 2 10')
        os.system('python dataloader/BatchDataloader.py  --env_name MazeEnv --min_max_num_agents 2 10 --mode all --over_write_preprocess False')
        previous_time = time.time()

    if (time.time() - start_time) > 172800:
        os.system('python datasetgenerator/DatasetGenMultiAgent.py --env_name BoxEnv --split_dataset True --min_max_num_agents 1 5')
        os.system('python dataloader/BatchDataloader.py  --env_name BoxEnv --min_max_num_agents 1 5 --mode all  --over_write_preprocess False')
        os.system('python datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv --split_dataset True --min_max_num_agents 2 10')
        os.system('python dataloader/BatchDataloader.py  --env_name MazeEnv --min_max_num_agents 2 10 --mode all  --over_write_preprocess False')
        break


# nohup python -u auto_split_convert_dataset.py  > log/auto_split_convert_dataset.txt 2>&1 &

# python dataloader/BatchDataloader.py  --env_name MazeEnv --min_max_num_agents 2 10 --mode all --over_write_preprocess False

# while True:
#     if (time.time() - previous_time) > 60*30:
#         os.system('python datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv --split_dataset True --min_max_num_agents 10 20 & '+
#                   'python datasetgenerator/DatasetGenMultiAgent.py --env_name BoxEnv --split_dataset True --min_max_num_agents 1 5 & wait')
#         os.system('python dataloader/BatchDataloader.py  --env_name MazeEnv --min_max_num_agents 10 20 --mode all --over_write_preprocess False & '+
#                   'python dataloader/BatchDataloader.py  --env_name BoxEnv --min_max_num_agents 1 5 --mode all  --over_write_preprocess False & wait')
#         previous_time = time.time()

#     if (time.time() - start_time) > 172800:
#         os.system('python datasetgenerator/DatasetGenMultiAgent.py --env_name MazeEnv --split_dataset True --min_max_num_agents 10 20 & '+
#                   'python datasetgenerator/DatasetGenMultiAgent.py --env_name BoxEnv --split_dataset True --min_max_num_agents 1 5 & wait')
#         os.system('python dataloader/BatchDataloader.py  --env_name MazeEnv --min_max_num_agents 10 20 --mode all  --over_write_preprocess False & '+
#                   'python dataloader/BatchDataloader.py  --env_name BoxEnv --min_max_num_agents 1 5 --mode all  --over_write_preprocess False & wait')
#         break

# nohup python -u auto_split_convert_dataset.py  > log/auto_split_convert_dataset.txt 2>&1 &