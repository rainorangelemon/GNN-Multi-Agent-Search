import time
import os
from pathlib import Path

start_time = time.time()
previous_time = 0
import glob

while True:
    print(f"{time.time() - start_time}")
    time.sleep(14400)
    os.system('kill -9 $(ps ax | grep "datasetgenerator/DatasetGenMultiAgent.py"  | awk \'{print $1}\')')
    os.system('bash multi_process_BoxEnv.sh')
    # os.system('bash multi_process_MazeEnv.sh')
    data_root =  Path('/media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/')

    data_path_maze = Path(data_root)/'MazeEnv'/f'dataset_3000_minNumAgent_1_maxNumAgent_5'/'Source'
    meta_files_maze = list(sorted(glob.glob(str(data_path_maze/f"*.graph")), reverse=True))
    
    list_ID_map = [int(item.split('ID_map_')[-1].split('_ID_graph')[0]) for item in data_path_maze]

    data_path_box = Path(data_root)/'BoxEnv'/f'dataset_3000_minNumAgent_1_maxNumAgent_5'/'Source'
    meta_files_box = list(sorted(glob.glob(str(data_path_box/f"*.graph")), reverse=True))
    
    list_ID_map_box = [int(item.split('ID_map_')[-1].split('_ID_graph')[0]) for item in meta_files_box]
    if (999 in list_ID_map) and (1999 in list_ID_map) and (2999 in list_ID_map) and (999 in list_ID_map_box) and (1999 in list_ID_map_box) and (2999 in list_ID_map_box):
        break


os.system('cd /media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/BoxEnv/; zip -r dataset_3000_minNumAgent_1_maxNumAgent_5.zip dataset_3000_minNumAgent_1_maxNumAgent_5')

# os.system('cd /media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/MazeEnv/; zip -r dataset_3000_minNumAgent_2_maxNumAgent_10.zip dataset_3000_minNumAgent_2_maxNumAgent_10')
