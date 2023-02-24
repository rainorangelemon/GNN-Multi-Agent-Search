import os
import os.path as osp
import shutil
from typing import Callable, List, Optional
import pickle
import numpy as np
import torch
import glob

import sys
sys.path.append("..")


import matplotlib.pyplot as pl
import matplotlib.patches as patches

from pathlib import Path
from easydict import EasyDict

from simulator.box_env import BoxEnv

Root_Path = '/media/qingbiao/Data/ql295/Data/MultiAgentSearch/'
dir_root_save = Path(Root_Path)

dir_dataset_map_save = dir_root_save/'dataset'/"BoxEnv"

def grep_map_id(num_agents):
    data_path = Path(data_root)/EnvName/f'dataset_3000_minNumAgent_{num_agents}_maxNumAgent_{num_agents}'
    print(data_path)
    dir_source_data = data_path / 'result_GNN' 
    meta_files = list(sorted(glob.glob(str(dir_source_data/f"*.gnn")),  key=lambda x: os.stat(x).st_size, reverse=False))
    ID_map_set = set()
    for meta_file in tqdm(meta_files):
        ID_map_set.add(int(meta_file.split('ID_map_')[-1].split('_ID_graph')[0]))
    return ID_map_set


def plot_path(config, min_num_agents, max_num_agents, num_agent, ID_map):

    config_setup = EasyDict(config)
    config_setup.min_num_agents = min_num_agents
    config_setup.max_num_agents = max_num_agents
    data_path = Path(data_root)/EnvName/f'dataset_3000_minNumAgent_{config_setup.min_num_agents}_maxNumAgent_{config_setup.max_num_agents}'

    dir_source_data = data_path / 'result_GNN' 
    
    ID_map, ID_graph, ID_CASE, numAgent = ID_map, 0, 0, num_agent
    
    path_file_gnn_sol_1_5 = dir_source_data/f'{EnvName}_ID_map_{ID_map:03d}_ID_graph_{ID_graph:03d}_ID_case_{ID_CASE:03d}_numAgent_{numAgent:03d}.gnn'
    path_file_graph_1_5 = data_path/'Source'/f'{EnvName}_ID_map_{ID_map:03d}_ID_graph_{ID_graph:03d}_ID_case_{ID_CASE:03d}_numAgent_{numAgent:03d}.graph'
    
    with open(path_file_gnn_sol_1_5, "rb") as file:
        data_dict_sol = pickle.load(file)
    with open(path_file_graph_1_5, "rb") as file:
        data_dict_graph = pickle.load(file)
        
    env = BoxEnv(config_setup)
    env.load_map(index=ID_map)
    env.init_new_problem_graph(index=ID_map,  graphs=data_dict_graph['graphs'])
    env.render_paths(data_dict_sol['solution'], (ID_map, ID_graph, ID_CASE),'GNN')
    # env.render_paths_line(data_dict_sol['solution'], (ID_map, ID_graph, ID_CASE),'GNN', data_dict_graph['goal_states'])

if __name__ == '__main__':
    config = {}

    config_BoxEnv = {'env_name': 'BoxEnv',
                    "exp_name": "AgentExplorerMultiAgent",

                    'data_root': '/media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/', 
                    "save_animation": "/media/qingbiao/Data/ql295/Data/MultiAgentSearch/",
                    'num_map_source': 3000,   

                    'k': 6,
                    'env_dim': 2,
                    "n_nodes": 1000,
                    "dim": 2,
                    'num_agents': 2,
                    "config_dim": 2,

                    "num_Env": 20,

                    "min_num_agents": 1,
                    "max_num_agents": 5,

                    'num_ProblemGraph_PerEnv': 5,
                    'num_Cases_PerProblemGraph': 1,

                    'split_train_end': 0.7,
                    'split_valid_end': 0.85,

                    "seed": 4123,
                    'timeout': 300,
                    }

    config_setup = EasyDict(config_BoxEnv)


    EnvName = 'BoxEnv'

    data_root = Path('/media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/')
    data_path = Path(data_root)/EnvName/f'dataset_3000_minNumAgent_{config_setup.min_num_agents}_maxNumAgent_{config_setup.max_num_agents}'

    dir_source_data = data_path / 'result_GNN' 

    meta_files_1_5 = list(sorted(glob.glob(str(dir_source_data/f"*.gnn")),  key=lambda x: os.stat(x).st_size, reverse=False))

    meta_files_w_1_005 = list(sorted(glob.glob(str(dir_source_data/f"*_w_1.005.gnn")),  key=lambda x: os.stat(x).st_size, reverse=False))

    from tqdm import tqdm

    ID_map_1_5 = set()
    for meta_file in tqdm(meta_files_1_5):
        ID_map_1_5.add(int(meta_file.split('ID_map_')[-1].split('_ID_graph')[0]))

    ID_map_1_5_w_1_005 = set()
    for meta_file in tqdm(meta_files_1_5):
        ID_map_1_5_w_1_005.add(int(meta_file.split('ID_map_')[-1].split('_ID_graph')[0]))


    num_agents = 6

    ID_map_6 = grep_map_id(num_agents)

    ID_map_8 = grep_map_id(8)

    # ID_map_10 = grep_map_id(10)
    ID_map_10 = {1948, 2426, 2480, 1943, 1961, 2976, 1961, 2454, 2942, 1439, 438, 2939, 1978}
    mutual_map = ID_map_1_5.intersection(ID_map_1_5_w_1_005).intersection(ID_map_6).intersection(ID_map_8).intersection(ID_map_10)

    print(mutual_map)
    # breakpoint()
    # {1934, 2443, 2444, 1427, 1428, 2446, 1429, 1430, 2957, 1431, 2449, 2958, 1432, 
    # 1941, 2450, 1942, 1434, 1943, 1944, 1437, 1438, 2965, 1439, 1948, 2966, 2968, 
    # 1951, 1442, 2969, 1443, 2461, 2970, 1953, 1445, 1446, 2464, 1447, 2465, 430, 
    # 2466, 1959, 1452, 436, 437, 1455, 1964, 2474, 1457, 2475, 440, 1458, 2477, 1460,
    #  1969, 1461, 1462, 2480, 2481, 1464, 1973, 2482, 2483, 1975, 449, 2485, 2994, 2486, 
    # 1978, 1470, 2997, 1981, 1474, 2492, 1475, 1476, 1477, 1478, 2496, 1479, 1480, 2498, 1481, 
    # 2499, 1482, 1993, 1485, 1488, 2926, 2927, 2929, 2930, 2931, 2932, 2933, 2934, 2425, 2426, 
    # 2937, 2939, 2940, 2427, 2428, 2429, 2433, 2432, 2942, 2943, 2944, 2945, 2951, 2947, 2949, 
    # 2952, 2955, 2956, 2445, 2953, 2950, 2960, 2961, 1938, 1426, 1939, 2451, 2453, 2455, 2962, 
    # 2454, 1945, 1947, 2963, 2964, 1946, 2456, 2457, 2458, 1950, 2459, 2468, 2460, 1952, 2463, 
    # 1955, 2473, 1956, 1448, 1957, 1449, 1958, 1450, 1960, 1961, 1963, 1965, 1967, 1459, 1968, 
    # 2478, 1971, 1972, 1465, 1974, 1466, 1467, 1468, 1979, 2487, 2488, 2489, 2490, 1982, 1983, 
    # 2494, 2495, 1986, 1987, 1988, 1989, 1990, 1991, 1483, 1992, 1484, 1994, 1995, 1996, 462, 
    # 1489, 1491, 1493, 1496, 1490, 1492, 1926, 1494, 1925, 1497, 1495, 1499, 1498, 2435, 1927, 
    # 2436, 1929, 1930, 2948, 2440, 1932, 2441, 1933, 2442}

    # min_num_agents = 1
    # max_num_agents = 5

    ID_map = 436

    # # ID_map = 1491
    plot_path(config_BoxEnv, min_num_agents=6,max_num_agents=6, num_agent=6, ID_map=ID_map)

    plot_path(config_BoxEnv, min_num_agents=8,max_num_agents=8, num_agent=8, ID_map=ID_map)

    plot_path(config_BoxEnv, min_num_agents=10,max_num_agents=10, num_agent=10, ID_map=1948)

