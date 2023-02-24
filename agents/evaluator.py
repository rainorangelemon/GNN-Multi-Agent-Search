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

from expert.CBS_GNN import GNNPlanner, pad_paths
from torch_geometric.data import Batch

class Evaluator(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        from dataloader.BatchDataloader import MyTestDataLoader
        from models.model import SpatialTemporalGNN

        self.model = SpatialTemporalGNN(embed_size=config.dim_embed, 
                                        temporal_encoding=config.temporal_encoding, 
                                        agent_identifier=config.agent_identifier)
        self.logger.info(f"Model: {self.model}\n")

        # define data_loader
        self.data_loader = MyTestDataLoader(config=config)

        # define optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)

        self.scheduler = None


        # define loss
        # self.loss = CrossEntropyLoss()

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0

        # TODO: pick a metric
        # -> discuss with Amanda, chenning
        # (success rate, - path length)

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
            self.model = self.model.to(self.config.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.config.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(checkpoint_dir=self.config.checkpoint_dir, epoch=self.config.test_epoch, lastest=self.config.lastest_epoch, best=self.config.best_epoch)

        # Summary Writer
        self.logger.info('*****New Simulator Enabled*****')

        self.summary_writer = wandb.init(project=config.exp_net, entity="multiagentsearch", config=config, dir=config.save_tb_data, name=config.exp_setting)

        wandb.run.log_code("../", include_fn=lambda path: 'model_multiAgent' in path or 'AgentExplorerMultiAgent_region' in path)

        # TODO: need to convertor
        # self.recorder = wandb.init(project=config.exp_name, entity="multiagentsearch", config=config)

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
        (data_path / 'result_GNN' / '').mkdir(parents=True, exist_ok=True)
        
        self.model.eval()

        dataloader = self.data_loader.test_loader

        for data in dataloader:

            starts = data['problem_setting'].starts_idx.data.cpu().numpy()
            goals = data['problem_setting'].goals_idx.data.cpu().numpy()

            # points, agent_ids, edge_index, heuristic, num_agents, num_nodes, goal_idxs
            predictor = Predictor(graph_data=data,
                                  dataset=dataloader.dataset,
                                  config = self.config,
                                  model=self.model,)

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
    
            if self.config.infer_w==float('inf'):
                gnn_path = data_path/ 'result_GNN' /f'{name_prefix}.gnn'
            else:
                gnn_path = data_path/ 'result_GNN' / f'{name_prefix}_w_{self.config.infer_w}.gnn'
            
            gnn_result = None
            if gnn_path.is_file():
                with open(gnn_path, 'rb') as file:
                    gnn_result = pickle.load(file)
            
            # if (gnn_result is None) or (not gnn_result['has_solution']):
            self.Env_Simulator = create_env_from_data(self.Env_Simulator, data, graphs=data_dict_expert['graphs'])

            try:
                t0 = time.perf_counter()
                planner = GNNPlanner()
                optimal_node = planner.plan(self.Env_Simulator, starts, goals, w=self.config.infer_w,
                                           model=predictor, time_budget=self.config.timeout, 
                                           plot_tree='GNN Tree')
                computation_time_model = time.perf_counter() - t0
            except IndexError as e:
                # impossible to have a solution, continue
                continue

            # save the gnn result
            with open(gnn_path, 'wb') as file:
                gnn_result = {
                    "ID_map": ID_map,
                    "ID_graph": ID_graph,
                    'ID_case': ID_case,
                    'num_agents': num_agents,
                    'has_solution': (optimal_node is not None),
                    'num_expand_nodes': len(planner.explored_nodes),
                    'solution': None if (optimal_node is None) else optimal_node.solution,
                    'flowtime': 0 if (optimal_node is None) else optimal_node.flowtime,
                    'makespan': 0 if (optimal_node is None) else optimal_node.makespan,
                    'computation_time': computation_time_model,
                }
                pickle.dump(gnn_result, file, protocol=pickle.HIGHEST_PROTOCOL)

            # log GNN solution
            self.summary_writer.log({
                'ID_map': gnn_result['ID_map'],
                'num_agents': gnn_result['num_agents'],
                'has_solution': gnn_result['makespan']!=0,
                'method': 'GNN',
                'num_expand_nodes': len(planner.explored_nodes),
                "makespan": gnn_result['makespan'],
                "flowtime": gnn_result['flowtime'],
                "computation_time": gnn_result['computation_time'],
            })
            
            # log CBS solution
            self.summary_writer.log({
                'ID_map': ID_map,
                'num_agents': num_agents,
                'has_solution': data_dict_expert['has_solution'],
                'method': 'CBS',
                "makespan": data_dict_expert['makespan'] if data_dict_expert['has_solution'] else 0,
                "flowtime": data_dict_expert['flowtime'] if data_dict_expert['has_solution'] else 0,
                "computation_time": data_dict_expert['computation_time'],
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


class Predictor:
    def __init__(self, graph_data, dataset, config, model, eval_h_range=20, alpha=1):

        self.device = config.device
        self.graph_data = graph_data
        self.dataset = dataset
        self.config = config
        self.model = model
        self.model.eval()

        self.eval_h_range = eval_h_range
        self.alpha = alpha

    @torch.no_grad()
    def give_path_prob(self, solution):
        data = self.dataset.solution2data(np.array(pad_paths(solution)), self.graph_data)
        data['p_node'].batch = torch.zeros(len(data['p_node'].x)).long()
        data['p_node'].ptr = torch.zeros(1).long()
        data = data.to(self.device)
        return self.model(data).data.cpu().numpy()[0]
        # return self.alpha*((self.model(data).data.cpu().numpy()[0]+self.eval_h_range).clip(0, 2*self.eval_h_range))
