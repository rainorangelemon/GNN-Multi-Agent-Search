import os
import logging
from pathlib import Path


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)


def return_dir_datasource(config):

    data_path_source = Path(config.data_root)/f'{config.env_name}' / f'dataset_{config.num_map_source}_minNumAgent_{config.min_num_agents}_maxNumAgent_{config.max_num_agents}'/'Source'
    data_path_source.mkdir(parents=True, exist_ok=True)

    return data_path_source

def return_dir_dataset_root(config):

    data_path_source = Path(config.data_root)/f'{config.env_name}' / f'dataset_{config.num_map_source}_minNumAgent_{config.min_num_agents}_maxNumAgent_{config.max_num_agents}'
    data_path_source.mkdir(parents=True, exist_ok=True)

    return data_path_source

def return_name_prefix_Envconfig(env_name, env_setting):
    (ID_MAP, ID_GRAPH, ID_CASE) = list(env_setting)[:3]

    name_prefix = f'{env_name}_ID_map_{ID_MAP:03d}_ID_graph_{ID_GRAPH:03d}_ID_case_{ID_CASE:03d}'

    return name_prefix

def return_name_prefix_Envconfig_NumAgent(env_name, env_setting, num_agents):
    (ID_MAP, ID_GRAPH, ID_CASE) = list(env_setting)[:3]

    name_prefix = f'{env_name}_ID_map_{ID_MAP:03d}_ID_graph_{ID_GRAPH:03d}_ID_case_{ID_CASE:03d}_numAgent_{num_agents:03d}'

    return name_prefix