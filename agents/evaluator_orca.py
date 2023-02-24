import torch
import torch.optim as optim

import wandb

import pickle
import shutil
import numpy as np

import time
from utils.config import print_cuda_statistics

from collections import defaultdict
from torch_sparse import SparseTensor
import os

from agents.base import BaseAgent
from simulator.maze_env import MazeEnv
from simulator.box_env import BoxEnv
from utils.dirs import return_name_prefix_Envconfig_NumAgent, return_dir_dataset_root

from scipy.special import expit
from expert.CBS import CBSPlanner, pad_paths
from torch_geometric.data import Batch
from expert.ORCA import ORCAPlanner


class EvaluatorORCA(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        from dataloader.BatchDataloader import MyTestDataLoader

        self.model = None
        self.scheduler = None

        # define data_loader
        self.data_loader = MyTestDataLoader(config=config)

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            self.config.device = torch.device("cuda:{}".format(self.config.gpu_device))
            torch.cuda.set_device(self.config.gpu_device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.config.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Summary Writer
        self.logger.info('*****New Simulator Enabled*****')

        self.summary_writer = wandb.init(project=config.exp_net, entity="multiagentsearch", config=config, dir=config.save_tb_data, name=config.exp_setting)

        wandb.run.log_code("../", include_fn=lambda path: 'model_multiAgent' in path or 'AgentExplorerMultiAgent_region' in path)

        self.Env_Simulator = eval(config.env_name)(config=self.config)

        # self.summary_writer.watch(self.model)
        self.time_record = None

    def run(self):
        """
        The main operator
        :return:
        """
        assert self.config.mode in ['train', 'test']
        try:
            print("-------test------------")
            start = time.perf_counter()
            self.test('test')
            self.time_record = time.perf_counter() - start

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    @torch.no_grad()
    def test(self, mode):
        """
        One cycle of model validation
        :return:
        """
        data_path = return_dir_dataset_root(self.config)
        (data_path / 'result_ORCA' / '').mkdir(parents=True, exist_ok=True)

        dataloader = self.data_loader.test_loader

        for data in dataloader:
            data = data.cpu()

            starts = data['problem_setting'].starts_idx.data.cpu().numpy()
            goals = data['problem_setting'].goals_idx.data.cpu().numpy()

            ID_map = int(data['env'].ID_map.cpu().numpy())
            ID_graph = int(data['env'].ID_graph.cpu().numpy())
            ID_case = int(data['env'].ID_CASE.cpu().numpy())
            num_agents = int(data['env'].num_agents.cpu().numpy())
            env_setting = (ID_map, ID_graph, ID_case)

            print('----------------------------------------------------')
            print('ID_map, ID_graph, ID_case, num_agents', ID_map, ID_graph, ID_case, num_agents)

            data_path = return_dir_dataset_root(self.config)
            name_prefix = return_name_prefix_Envconfig_NumAgent(self.config.env_name, env_setting, num_agents)
            path_to_expertSol = data_path/ 'Source' /f'{name_prefix}.graph'

            with open(path_to_expertSol, 'rb') as file:
                data_dict_expert = pickle.load(file)

            self.Env_Simulator = create_env_from_data(self.Env_Simulator, data, graphs=data_dict_expert['graphs'])
            try:
                t0 = time.perf_counter()
                optimal_node = ORCAPlanner.plan(self.Env_Simulator, starts, goals,  env_setting=env_setting, time_budget=self.config.timeout, animation=False)
                computation_time_model = time.perf_counter() - t0
            except IndexError as e:
                # impossible to have a solution, continue
                print('Impossible to have a solution.. conitnue')
                continue
            
            # save the optimal node
            if optimal_node is not None:
                with open(data_path/ 'result_ORCA' /f'{name_prefix}.orca', 'wb') as file:
                    orca_result = {
                        "ID_map": ID_map,
                        "ID_graph": ID_graph,
                        'ID_case': ID_case,
                        'num_agents': num_agents,
                        'solution': optimal_node.solution,
                        'flowtime': optimal_node.flowtime,
                        'makespan': optimal_node.makespan,
                        'computation_time': computation_time_model,
                    }
                    pickle.dump(orca_result, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            # log ORCA solution
            self.summary_writer.log({
                'ID_map': ID_map,
                'num_agents': num_agents,
                'has_solution': (optimal_node is not None),
                'method': 'ORCA',
                "makespan": 0 if (optimal_node is None) else optimal_node.makespan,
                "flowtime": 0 if (optimal_node is None) else optimal_node.flowtime,
                "computation_time": computation_time_model,
            })

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        if self.config.mode == 'train':
            print(self.model)
        print("Experiment on {} finished.".format(self.config.exp_name))
        print("Please wait while finalizing the operation.. Thank you")
        # self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        self.data_loader.finalize()
        if self.config.mode == 'test':
            print("################## End of testing ################## ")
            print("Computation timeheuristic:\t{} ".format(self.time_record))


def create_env_from_data(env, data, graphs):
    data = data.detach().cpu().clone()
    env.init_new_problem_graph(index=int(data['env'].ID_map.numpy().reshape(-1)), graphs=graphs)
    return env