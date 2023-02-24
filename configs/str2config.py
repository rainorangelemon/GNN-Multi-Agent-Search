from configs import BoxEnv_region, MazeEnv_region
import argparse
import socket
import os
from copy import deepcopy
currentdir = os.path.dirname(os.path.realpath(__file__))

class UserNamespace(object):
    pass


def generate_data_paths():
    ip_address = socket.gethostbyname(socket.gethostname())

    # if 'ql295' in currentdir and ip_address == '128.232.69.16':
    #   config_preprocess = {**{"data_root": "/net/archive/export/ql295/Data/MultiAgentSearch/dataset/"}, **config_preprocess}
    #   config_experiment = {**{"data_root": "/net/archive/export/ql295/Data/MultiAgentSearch/dataset/",
    #   "save_data": "/net/archive/export/ql295/Data/MultiAgentSearch/",
    #   "save_animation": "/net/archive/export/ql295/Data/results/animation/",
    #   "save_tb_data": "/net/archive/export/ql295/Data/wandb"}, **config_experiment}

    base_preprocess = {
        'data_root': '/dataset/',
    }

    base_experiment = {
        'data_root': '/dataset/',
        'save_data': '/',
        'save_animation': '/results/animation/',
        'save_tb_data': '/wandb'
    }

    if 'ql295' in currentdir and ip_address == '127.0.1.1' :
        base_path = '/local/scratch/ql295/Data/MultiAgentSearch'
        config_preprocess_path = {k: base_path+v for k, v in base_preprocess.items()}
        config_experiment_path = {k: base_path+v for k, v in base_experiment.items()}

    elif 'rainorangelemon' in currentdir and 'home' in currentdir:
        base_path = "/home/rainorangelemon/Documents/GNN_MultiAgentSearch"
        config_preprocess_path = {k: base_path+v for k, v in base_preprocess.items()}
        config_experiment_path = {k: base_path+v for k, v in base_experiment.items()}

    elif 'rainorangelemon' in currentdir and 'Users' in currentdir:
        base_path = "/Users/rainorangelemon/Documents/GNN_MultiAgentSearch"
        config_preprocess_path = {k: base_path + v for k, v in base_preprocess.items()}
        config_experiment_path = {k: base_path + v for k, v in base_experiment.items()}

    elif 'ql295' in currentdir and ip_address == '128.232.69.16':
        base_path = '/local/scratch/ql295/Data/MultiAgentSearch'
        base_path_Data = '/local/scratch/ql295/Data/MultiAgentSearch/dataset'
        # base_path = "/net/archive/export/ql295/Data/MultiAgentSearch"
        # base_path_Data = "/net/archive/export/ql295/Data"
        config_preprocess_path = {k: base_path+v for k, v in base_preprocess.items()}
        config_experiment_path = {k:base_path+v if k in ['data_root', 'save_data'] else base_path_Data+v for k, v in base_experiment.items()}        

    elif 'hpc-work' in currentdir:
        base_path = "/home/ql295/rds/hpc-work/GNN_MultiAgentSearch"
        base_path_Data = "/home/ql295/rds/hpc-work/GNN_MultiAgentSearch/Data"
        config_preprocess_path = {k: base_path+v for k, v in base_preprocess.items()}
        config_experiment_path = {k:base_path+v if k in ['data_root', 'save_data'] else base_path_Data+v for k, v in base_experiment.items()}

    elif 'qingbiao' in currentdir:
        base_path = "/media/qingbiao/Data/ql295/Data/MultiAgentSearch/"
        base_path_Data = "/media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/"
        config_preprocess_path = {k: base_path+v for k, v in base_preprocess.items()}
        config_experiment_path = {k:base_path+v if k in ['data_root', 'save_data'] else base_path_Data+v for k, v in base_experiment.items()}


    else:
        base_path = "/home/qingbiao/PycharmProjects/GNN_MultiAgentSearch"
        config_preprocess_path = {k: base_path+v for k, v in base_preprocess.items()}
        config_experiment_path = {k: base_path+v for k, v in base_experiment.items()}

    return {'preprocess': config_preprocess_path, 
            'train': deepcopy(config_experiment_path), 
            'test': deepcopy(config_experiment_path),}    


str2config = {
    'BoxEnv': {'preprocess': BoxEnv_region.config_preprocess, 
               'train': deepcopy(BoxEnv_region.config_experiment),
               'test': deepcopy(BoxEnv_region.config_experiment)},
    'MazeEnv': {'preprocess': MazeEnv_region.config_preprocess,
                'train': deepcopy(MazeEnv_region.config_experiment),
                'test': deepcopy(MazeEnv_region.config_experiment)},
}

path_dict = generate_data_paths()
for env, cfgs in str2config.items():
    for k, v in cfgs.items():
        if 'rainorangelemon' in path_dict['train']['data_root']:
            v['exp_net'] = 'chenning'

        if k in ['train', 'test']:
            v['mode'] = k
        if k=='train':
            v['agent'] = 'Trainer'
        if k=='test':
            v['pin_memory'] = False
            v['agent'] = 'Evaluator'
            v['negative_sampling'] = ''
        cfgs[k] = {**path_dict[k], **v}


def add_default_argument_and_parse(parser, keyword):
    user_namespace = UserNamespace()
    parser.add_argument("--env_name", type=str, default='BoxEnv')
    if keyword == 'experiment':  
        parser.add_argument("--mode", type=str, default='train')
        parser.parse_known_args(namespace=user_namespace)
        default_config = str2config[user_namespace.env_name][user_namespace.mode]
    else:
        parser.parse_known_args(namespace=user_namespace)
        default_config = str2config[user_namespace.env_name][keyword]
    for k, v in default_config.items():
        if isinstance(v, list):
            parser.add_argument(f"--{k}", nargs=len(v), type=type(v[0]), default=v)
        elif k not in vars(user_namespace):
            parser.add_argument(f"--{k}", type=type(v), default=v)            
    parser.parse_known_args(namespace=user_namespace)
    user_namespace.min_num_agents, user_namespace.max_num_agents = int(user_namespace.min_max_num_agents[0]), int(user_namespace.min_max_num_agents[1])
    return user_namespace