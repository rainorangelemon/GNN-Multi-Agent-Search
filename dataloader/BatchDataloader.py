import os
import random
import sys
import os.path as osp
import shutil
from typing import Callable, List, Optional
import pickle
import numpy as np
import torch

import glob
import logging
import time

from easydict import EasyDict
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from models.model import PositionalEncoding1D
from collections import defaultdict
from tqdm import tqdm
from expert.CBS import pad_paths


currentdir = os.path.dirname(os.path.realpath(__file__))

parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from pathlib import Path

from utils.dirs import return_dir_datasource, return_dir_dataset_root, return_name_prefix_Envconfig, return_name_prefix_Envconfig_NumAgent

torch.multiprocessing.set_sharing_strategy('file_system')


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


class MyTrainDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"Dataset from environment:  {config.env_name}")
        log_info = f"Dataset generated from {config.num_agents} agents  environments \
            with {config.num_ProblemGraph_PerEnv} problem graphs per environment and {config.num_Cases_PerProblemGraph} cases per problem graph."
        self.logger.info(log_info)
        self.create_loaders()

    def create_loaders(self):
        self.start_time = time.time()
        data_path = return_dir_dataset_root(self.config)
        
        train_dataset = CBSNodeDataset(self.config, str(data_path / 'train'), 'train')

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=self.config.shuffle_dataset,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory,
                                       collate_fn=collate)
        
        valid_dataset = CBSNodeDataset(self.config, str(data_path / 'valid'), 'valid')
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.config.valid_batch_size,
                                       shuffle=False,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory,
                                       collate_fn=collate)
        
        self.loaders = {'cbs': self.train_loader}
        self.iterloaders = {k: iter(v) for k, v in self.loaders.items()}

#     def get_next_data_for_each_loader(self, key):
#         if (time.time()-self.start_time) > 1800:
#             self.create_loaders()
#         try:
#             batch = next(self.iterloaders[key])
#         except StopIteration:
#             self.iterloaders[key] = iter(self.loaders[key])
#             batch = next(self.iterloaders[key])
#         return batch

#     def get_next_data(self):
#         return self.get_next_data_for_each_loader('cbs')

    def finalize(self):
        pass


class MyTestDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(f"Dataset from environment:  {config.env_name}")
        log_info = f"Dataset generated from {config.num_agents} agents  environments \
            with {config.num_ProblemGraph_PerEnv} problem graphs per environment and {config.num_Cases_PerProblemGraph} cases per problem graph."
        self.logger.info(log_info)

        data_path = return_dir_dataset_root(config)
        
        self.dataset = SolvableGraphDataset(config, str(data_path / 'test'), 'test')

        self.test_loader = DataLoader(self.dataset,
                                       batch_size=self.config.test_batch_size,
                                       shuffle=False,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)

    def finalize(self):
        pass


class MyGraphDataset(Dataset):
    def __init__(self, config, root, mode, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.te_encode = PositionalEncoding1D(config.dim_te)
        self.dim_embed = config.dim_embed
        self.data_path = Path(root)
        self.config = config
        self.dir_rawdata = self.data_path/'raw'
        self.mode = mode
        self.Env_name = self.data_path.parent.parent.name

        print(f"Dataset mode: {self.data_path.name} and environment: {self.Env_name}")
        
        self.logger = logging.getLogger(f"Dataset from environment:  {self.Env_name}")
        self.logger.info(f"Mode: {self.data_path.name}")

        self.optimal_files = list(sorted(glob.glob(str(self.dir_rawdata / "*.optimal"))))
        self.graph_files = list(sorted(glob.glob(str(self.dir_rawdata / "*.graph"))))

    def _process_graph(self, file_path):

        data = HeteroData()

        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)

        ID_MAP = int(file_path.split('ID_map_')[-1].split('_ID_graph')[0])
        ID_graph = int(file_path.split('_ID_graph_')[-1].split('_ID_case')[0])
        ID_CASE = int(file_path.split('_ID_case_')[-1].split('_numAgent')[0])
        num_agents = int(file_path.split('numAgent_')[-1].split('.graph')[0])

        points = np.array(data_dict['graphs']['points'])
        edge_index = np.array(data_dict['graphs']['edge_index'])
        # TODO: why it is zero?
        data['g_node'].x = torch.FloatTensor(points)
        data['g_node'].pos = torch.FloatTensor(points)
        data['g_node'].num_nodes = len(points)
        data['g_node', 'g_to_g', 'g_node'].edge_index = torch.LongTensor(edge_index.T)
        #TODO: distance between different points in graph 
        data['g_node', 'g_to_g', 'g_node'].edge_attr = data['g_node'].pos[edge_index.T[0, :], :] - data['g_node'].pos[edge_index.T[1, :], :]


        data['env'].ID_map = torch.LongTensor([ID_MAP])
        data['env'].ID_graph = torch.LongTensor([ID_graph])
        data['env'].ID_CASE = torch.LongTensor([ID_CASE])
        data['env'].num_agents = torch.LongTensor([num_agents])
        data['has_solution'] = torch.BoolTensor([data_dict['has_solution']])

        data['problem_setting'].starts_idx = torch.LongTensor(data_dict['init_states'])
        data['problem_setting'].goals_idx = torch.LongTensor(data_dict['goal_states'])
        data['problem_setting'].num_nodes = num_agents
        data['computation_time'] = torch.from_numpy(np.array(data_dict['computation_time']))
        data['num_cbs_nodes'] = torch.from_numpy(np.array(len(data_dict['Vs'])))

        assert len(data['problem_setting'].goals_idx)==int(data['env'].num_agents)

        return data, data_dict['Vs']

    def _process_solution(self, file_path, graph_data):
        assert 'cbsnode' in file_path
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)

        solution = np.array(pad_paths(data_dict['solution']))
        assert len(solution.shape)==2
        assert len(solution)==int(graph_data['env'].num_agents)

        data = self.solution2data(solution, graph_data)
        data.V = torch.LongTensor(np.array([data_dict['V']]))
        return data

    def solution2data(self, solution, graph_data):
        data = graph_data.clone()

        num_agents, makespan = solution.shape
        ids = np.tile(np.arange(num_agents).reshape(-1, 1), (1, makespan))
        timesteps = np.tile(np.arange(makespan).reshape(1, -1), (num_agents, 1))
        # N * T * MP
        # 'p_node': agents' solution paths
        data['p_node'].path = torch.from_numpy(
            np.concatenate((ids[..., None], timesteps[..., None], solution[..., None]), axis=-1).reshape(-1, 3)).long()
        data['p_node'].x = self.te_encode(data['p_node'].path[:, 1])

        data['p_node'].num_nodes = len(data['p_node'].path)

        # 'g_node': sampled nodes in the graph
        data['p_node', 'p_to_g', 'g_node'].edge_index = torch.cat(
            (torch.arange(len(data['p_node'].path)).reshape(1, -1),
             data['p_node'].path[:, 2].reshape(1, -1)), dim=0)
        data['g_node', 'g_to_p', 'p_node'].edge_index = data['p_node', 'p_to_g', 'g_node'].edge_index.clone().flip(0)

        # (t-1 -> t) edges
        adjacent_edges = torch.cat(
            (torch.arange(1, len(data['p_node'].path)).reshape(1, -1),
             torch.arange(len(data['p_node'].path) - 1).reshape(1, -1)), dim=0)
        # (t-1 -> t) and (t -> t-1)
        adjacent_edges = torch.cat((adjacent_edges, adjacent_edges.flip(0)), dim=-1)
        # filter out the edge with same agent id
        adjacent_edges = adjacent_edges[:, data['p_node'].path[adjacent_edges[0, :], 0] == data['p_node'].path[adjacent_edges[1, :], 0]]

        # inter-agent edge: full connect if sharing the same time step
        x, y = torch.arange(len(data['p_node'].path)), torch.arange(len(data['p_node'].path))
        grid_x, grid_y = torch.meshgrid(x, y)
        candidate_edge = torch.cat((grid_x.reshape(1, -1), grid_y.reshape(1, -1)), dim=0)
        # filter out self-loop
        candidate_edge = candidate_edge[:, candidate_edge[0,:]!=candidate_edge[1,:]]
        # True if sharing the same time
        is_candidate = data['p_node'].path[candidate_edge[0,:], 1]==data['p_node'].path[candidate_edge[1,:], 1]
        temporal_edge = candidate_edge[:, is_candidate]

        # concatenate adjacent_edges and temporal_edges
        data['p_node', 'p_to_p', 'p_node'].edge_index = edge_index = torch.cat((adjacent_edges, temporal_edge), dim=-1)
        # return the vertex index of the graph
        g_id_on_edge = data['p_node'].path[edge_index.reshape(-1), 2].reshape(edge_index.shape)
        # relative geometric offset between nodes
        edge_attr = data['g_node'].pos[g_id_on_edge[0, :], :] - data['g_node'].pos[g_id_on_edge[1, :], :]
        # create the one-hot label, [1, 0] for adjacent_edge, [0, 1] for temporal_edge
        edge_attr = torch.cat((torch.zeros(len(edge_attr), 2), edge_attr), dim=-1)
        edge_attr[:adjacent_edges.shape[1], 0] = 1
        edge_attr[adjacent_edges.shape[1]:, 1] = 1
        data['p_node', 'p_to_p', 'p_node'].edge_attr = edge_attr

        return data


class CBSNodeDataset(MyGraphDataset):
    # return the solution that has V=0 for solved cases

    def __init__(self, config, root, mode, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(config, root, mode, transform, pre_transform, pre_filter)
        self.pair_data = []       
        for file_path in tqdm(self.graph_files):
            if file_path.replace('.graph', '.optimal') not in self.optimal_files:
                continue
            with open(file_path, 'rb') as file:
                data_dict = pickle.load(file)
            Vs = data_dict['Vs']
            if (0 not in Vs):
                print(file_path)
            if (np.std(Vs) == 0) or (0 not in Vs):
                continue
            good_Vid = [vid for vid, V in enumerate(Vs) if V==0]
            good_cbs_files = [file_path.replace('.graph', f'.cbsnode{vid}') for vid in good_Vid]
            good_depths = []
            for path in good_cbs_files:
                with open(path, 'rb') as f:
                    data_dict = pickle.load(f)
                    good_depths.append(data_dict['depth'])
            
            bad_Vid = [vid for vid, V in enumerate(Vs) if V!=0]
            bad_cbs_files = [file_path.replace('.graph', f'.cbsnode{vid}') for vid in bad_Vid]
            bad_depths = []
            for path in bad_cbs_files:
                with open(path, 'rb') as f:
                    data_dict = pickle.load(f)
                    bad_depths.append(data_dict['depth'])
            
            for good_depth, good_cbs_file in zip(good_depths, good_cbs_files):
                self.pair_data.extend([[good_cbs_file, bad_cbs_file, file_path] for bad_depth, bad_cbs_file in zip(bad_depths, bad_cbs_files) if (bad_depth<=good_depth)])
                
        print(len(self.pair_data))
                
    def len(self):
        return len(self.pair_data)

    def get(self, index):
        good_cbs_file, bad_cbs_file, file_path = self.pair_data[index]
        
        graph_data, Vs = self._process_graph(file_path)
        good_data = self._process_solution(good_cbs_file, graph_data)
        
        graph_data, Vs = self._process_graph(file_path)
        bad_data = self._process_solution(bad_cbs_file, graph_data)

        return good_data, bad_data


class SolvableGraphDataset(MyGraphDataset):
    def __init__(self, config, root, mode, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(config, root, mode, transform, pre_transform, pre_filter)
        # train_files = list(sorted(glob.glob(str(self.data_path / '..' / 'train' / 'raw' / "*.graph"))))
        # for file_path in tqdm(train_files):
        #     with open(file_path, 'rb') as file:
        #         data_dict = pickle.load(file)
        #     Vs = data_dict['Vs'] 
        #     if (0 not in Vs):  # no interest on the simple cases
        #         self.graph_files.append(file_path)
        
    def len(self):
        return len(self.graph_files)

    def get(self, index):
        graph_data, _ = self._process_graph(self.graph_files[index])
        return graph_data
