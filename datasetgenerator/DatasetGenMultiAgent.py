
import glob
import shutil

import os
import sys
import time
import pickle
import argparse

import tracemalloc
from guppy import hpy

from pathlib import Path

currentdir = os.path.dirname(os.path.realpath(__file__))

parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.config import seed_everything
from expert.CBS import CBSPlanner
import numpy as np
import copy
from utils.dirs import return_dir_datasource, return_name_prefix_Envconfig, return_name_prefix_Envconfig_NumAgent
from tqdm import tqdm
from multiprocessing import Pool  

from configs.str2config import add_default_argument_and_parse
import gc

from simulator.maze_env import MazeEnv
from simulator.box_env import BoxEnv

parser = argparse.ArgumentParser(description="Generation Dataset From CBS")
config_setup = add_default_argument_and_parse(parser, 'preprocess')


def compute_thread(config_setup, env_setting):
    try:
        # print(thread_id)
        # id_case = self.task_queue.get(block=False)
        # print('thread {} get task:{}'.format(thread_id, id_case))
        runExpertSolver(config_setup, env_setting)
        # print('thread {} finish task:{}'.format(thread_id, id_case))
    except Exception as e:
        print(e)
        # raise e
    finally:
        gc.collect()


def runExpertSolver(config_setup, env_setting):
    ID_map, ID_graph, ID_case, label = env_setting

    seed_everything((1+ID_map)*(1+ID_graph)*(1+ID_case))
    env = eval(config_setup.env_name)(config=config_setup)
    env.init_new_problem_graph(ID_map)
    env.init_new_problem_instance(ID_case)

    name_prefix = return_name_prefix_Envconfig(config_setup.env_name, env_setting)
    # print(f'num_agents changed into {config_setup.num_agents} for {ID_map} Map {ID_graph} Graph {ID_case} Case')

    print(f'{name_prefix}_numAgent_{config_setup.num_agents}.graph')
    data_path = return_dir_datasource(config_setup)
    name_file = data_path/f'{name_prefix}.graph'
    if name_file.is_file():
        return

    start = time.perf_counter()
    planner = CBSPlanner()
    optimal_node, cbs_nodes = planner.plan(env, env.agent_starts_vidxs, env.agent_goals_vidxs, need_update_V=True, time_budget=config_setup.timeout)
    time_record = time.perf_counter() - start

    file_paths = []

    graph_data = create_graph_data(env, env_setting, planner, optimal_node, cbs_nodes, time_record)
    file_paths.append(save_solution(config_setup, graph_data, data_path, env_setting, 'graph'))

    if optimal_node is not None:
        optimal_data = create_optimal_data(env_setting, optimal_node)
        file_paths.append(save_solution(config_setup, optimal_data, data_path, env_setting, 'optimal'))

    for idnode, node in enumerate(cbs_nodes):
        node_data = create_cbs_node_data(env_setting, node)
        file_paths.append(save_solution(config_setup, node_data, data_path, env_setting, f'cbsnode{idnode}'))

    if optimal_node is not None:
        del optimal_node
    if cbs_nodes is not None:
        del cbs_nodes

    if config_setup.split_during_generate:
        move_all_files(config_setup, [label]*len(file_paths), file_paths)


def create_graph_data(env, env_setting, planner, optimal_node, cbs_nodes, time_record):
    ID_map, ID_graph, ID_case, label = env_setting

    data = {
        "ID_map": ID_map,
        "ID_graph": ID_graph,
        'ID_case': ID_case,
        'has_solution': (optimal_node is not None),
        'num_expand_nodes': len(planner.explored_nodes),
        'computation_time': time_record,
        "graphs": {k: v for k, v in env.graphs.items()},
        "Vs": [node.V for node in cbs_nodes],
        "init_states": env.agent_starts_vidxs,
        "goal_states": env.agent_goals_vidxs,
    }

    if optimal_node is not None:
        data['flowtime'] = optimal_node.flowtime,
        data['makespan'] = optimal_node.makespan

    return data


def create_optimal_data(env_setting, optimal_node):
    ID_map, ID_graph, ID_case, label = env_setting
    data = {
        "ID_map": ID_map,
        "ID_graph": ID_graph,
        'ID_case': ID_case,
        'solution': optimal_node.solution,
    }

    return data


def create_cbs_node_data(env_setting, cbs_node):
    ID_map, ID_graph, ID_case, label = env_setting

    data = {
        "ID_map": ID_map,
        "ID_graph": ID_graph,
        'ID_case': ID_case,
        'solution': cbs_node.solution,
        'V': cbs_node.V,
        'depth': cbs_node.depth,
    }

    return data


def save_solution(config_setup, Data_to_stored, data_path, env_setting, suffix):
    
    # name_prefix = return_name_prefix_Envconfig(config_setup.env_name, env_setting)
    name_prefix = return_name_prefix_Envconfig_NumAgent(config_setup.env_name, env_setting, config_setup.num_agents)
    name_file = data_path/f'{name_prefix}.{suffix}'
    
    with open(name_file, "wb") as file:
        meta = pickle.dump(Data_to_stored, file, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f'Saved dataset at {name_file} \n')

    return name_file


def split_dataset(config_setup):
    data_path = return_dir_datasource(config_setup)
    meta_files_full = list(glob.glob(str(data_path/"*.pickle")))
    meta_files = list(sorted(meta_files_full))
    labels = [label_task(config_setup, id_file, 0, len(meta_files)) for id_file in range(len(meta_files))]

    move_all_files(config_setup, labels, meta_files)


def move_all_files(config_setup, labels, meta_files):
    data_path = return_dir_datasource(config_setup)
    label2dir = {key: data_path.parent / key / 'raw' for key in ['train', 'valid', 'test']}
    for target in label2dir.values():
        target.mkdir(parents=True, exist_ok=True)
    for id_file, label in tqdm(zip(range(len(meta_files)), labels)):
        name_file = os.path.basename(meta_files[id_file])
        target_file = str(label2dir[label] / name_file)
        move_file(meta_files[id_file], target_file)


def move_file(origin, target):
    shutil.copy(origin, str(target))
    # print(f'Move file from \n\t{origin}\n\t to \n\t {target}.')


def label_task(config_setup, id_map, start, end):
    if config_setup.mode == 'all':
        num_map = end - start
        end_train = int(config_setup.split_train_end * num_map)
        end_val = int(config_setup.split_valid_end * num_map)
        return 'train' if (id_map - start) < end_train else \
               'valid' if (id_map - start) < end_val else \
               'test'
    else:
        return config_setup.mode


if __name__ == '__main__':

    print(config_setup)
    seed_everything(config_setup.seed)

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    with Pool(config_setup.n_workers) as pool:

        tasks = [(ID_map, ID_graph, ID_case, label_task(config_setup, ID_map, config_setup.ID_Env_Start, config_setup.ID_Env_End)) \
                 for ID_map in range(config_setup.ID_Env_Start, config_setup.ID_Env_End) \
                 for ID_graph in range(config_setup.num_ProblemGraph_PerEnv) \
                 for ID_case in range(config_setup.num_Cases_PerProblemGraph)]
        tasks_is_contained = []
        task_setups = []
        for task in tasks:
            config_setup_copy = copy.deepcopy(config_setup)
            config_setup_copy.num_agents = np.random.randint(config_setup_copy.min_num_agents, 1+config_setup_copy.max_num_agents)

            env_setting = task
            ID_map, ID_graph, ID_case, _ = task
            task_setups.append((config_setup_copy, env_setting))

            name_prefix = return_name_prefix_Envconfig_NumAgent(config_setup.env_name, env_setting, config_setup_copy.num_agents)
            data_path = return_dir_datasource(config_setup)
            name_file = data_path/f'{name_prefix}.graph'

            if name_file.is_file():
                with open(str(name_file), 'rb') as file:
                    try:
                        data_dict = pickle.load(file)
                        print(f'{name_file} already exists. Skip this in the task pool')
                        tasks_is_contained.append(True)
                        continue
                    except Exception as e:
                        print(f"Error loading file: {name_file}. Remove it.")
                        os.system(f'rm {name_file}')
            tasks_is_contained.append(False)
            print(f'num_agents changed into {config_setup_copy.num_agents} for {ID_map} Map {ID_graph} Graph {ID_case} Case')

        if np.any(tasks_is_contained):
            iterable = task_setups[(np.argmax(np.where(np.array(tasks_is_contained))[0])+1):]
        else:
            iterable = task_setups

    for task_setup in tqdm(iterable):
        compute_thread(*task_setup)
        h = hpy()
        print(h.heap())
        print(h.heap().byrcs)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        #     executor.submit(compute_thread, *task_setup)

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 10 differences ]")
    for stat in top_stats:#[:10]:
        print(stat)