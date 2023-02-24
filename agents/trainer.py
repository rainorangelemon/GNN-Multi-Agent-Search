import torch
import torch.optim as optim

import wandb

import pickle
import shutil
import numpy as np
from tqdm import tqdm

import time
from utils.config import print_cuda_statistics

from collections import defaultdict
from simulator.Visualizer import plot_v_and_h
from torch_sparse import SparseTensor
import os

from agents.base import BaseAgent
from simulator.maze_env import MazeEnv
from simulator.box_env import BoxEnv
from utils.dirs import return_name_prefix_Envconfig_NumAgent, return_dir_dataset_root

from scipy.special import expit
from expert.CBS import CBSPlanner


class Trainer(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        from dataloader.BatchDataloader import MyTrainDataLoader

        from models.model import SpatialTemporalGNN

        self.model = SpatialTemporalGNN(embed_size=config.dim_embed,
                                        temporal_encoding=config.temporal_encoding, 
                                        agent_identifier=config.agent_identifier)
        self.logger.info(f"Model: {self.model}\n")

        # define data_loader
        self.data_loader = MyTrainDataLoader(config=config)

        # define optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)

        if config.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.max_epoch, eta_min=1e-6)
        else:
            self.scheduler = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0

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
        if self.config.con_train:
            self.load_checkpoint(checkpoint_dir=self.config.checkpoint_dir_load, epoch=self.config.test_epoch, lastest=self.config.lastest_epoch, best=self.config.best_epoch)

        else:
            self.load_checkpoint(checkpoint_dir=self.config.checkpoint_dir, epoch=self.config.test_epoch, lastest=self.config.lastest_epoch, best=self.config.best_epoch)

        # Summary Writer
        self.logger.info('*****New Simulator Enabled*****')

        self.summary_writer = wandb.init(project=config.exp_net, entity="multiagentsearch", config=config, dir=config.save_tb_data, name=config.exp_setting)
        wandb.run.log_code("../", include_fn=lambda path: 'trainer' in path or 'BatchDataLoader' in path)
        self.Env_Simulator = eval(config.env_name)(config=self.config)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        best_valid_loss = float('inf')

        for epoch in range(self.current_epoch, self.config.max_epoch + 1):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.logger.info(f'Train at Epoch {self.current_epoch}: Learning Rate: {self.optimizer.param_groups[0]["lr"]:.1E}')

            self.save_checkpoint(epoch, lastest=True)
            if (epoch + 1) % self.config.validate_every == 0:
                valid_loss = self.validate()
                self.logger.info(f'Validate at Epoch {self.current_epoch}: Valid Loss: {valid_loss}')
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.save_checkpoint(epoch, is_best=True, lastest=False)

                if self.scheduler is not None:
                    self.scheduler.step()

            
    def get_loss_pair(self, pair_data1, pair_data2):
        h_pair_data1 = self.model(pair_data1)
        h_pair_data2 = self.model(pair_data2)

        # how this part works?
        gap_1_better_than_2 = (-h_pair_data2 + h_pair_data1 + self.config.train_gamma).relu()
        num_1_better_than_2 = (pair_data1.V < pair_data2.V)
        loss_1_better_than_2 = (gap_1_better_than_2 * num_1_better_than_2) / (1e-9 + num_1_better_than_2.sum())

        gap_2_better_than_1 = (-h_pair_data1 + h_pair_data2 + self.config.train_gamma).relu()
        num_2_better_than_1 = (pair_data2.V < pair_data1.V)
        loss_2_better_than_1 = (gap_2_better_than_1 * num_2_better_than_1) / (1e-9 + num_2_better_than_1.sum())

        contrastive_loss = loss_1_better_than_2.sum() + loss_2_better_than_1.sum()
       
        return contrastive_loss, h_pair_data1, h_pair_data2
            
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        # Set the model to be in training mode
        self.model.train()
        log_record = defaultdict(list)

        for pair_data1, pair_data2 in self.data_loader.train_loader:

            pair_data1 = pair_data1.to(self.config.device)
            pair_data2 = pair_data2.to(self.config.device)

            loss, h_pair_data1, h_pair_data2 = self.get_loss_pair(pair_data1, pair_data2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            log_key2str = {'loss/train': loss.item()}

            for key, value in log_key2str.items():
                log_record[key].append(value)

            if self.current_iteration % self.config.log_loss_interval == 0:

                self.summary_writer.log({**{"Iteration": self.current_iteration},
                                         **{k: np.mean(np.asarray(v)) for k, v in log_record.items()}})

                log_record = defaultdict(list)

            if self.current_iteration % (self.config.log_visual_interval) == 0:          
                
                img_constrastive = self.draw_h(pair_data1, pair_data2, h_pair_data1, h_pair_data2)        
                self.summary_writer.log({
                    "img/train": wandb.Image(img_constrastive),
                })
                
            self.current_iteration += 1
            
    @torch.no_grad()
    def validate(self):
        # Set the model to be in training mode
        self.model.eval()
        log_record = defaultdict(list)

        for pair_data1, pair_data2 in tqdm(self.data_loader.valid_loader):

            pair_data1 = pair_data1.to(self.config.device)
            pair_data2 = pair_data2.to(self.config.device)

            loss, h_pair_data1, h_pair_data2 = self.get_loss_pair(pair_data1, pair_data2)
        
            log_key2str = {'loss/valid': loss.item()}

            for key, value in log_key2str.items():
                log_record[key].append(value)

        self.summary_writer.log({k: np.mean(np.asarray(v)) for k, v in log_record.items()})
        img_constrastive = self.draw_h(pair_data1, pair_data2, h_pair_data1, h_pair_data2)        
        self.summary_writer.log({
            "img/valid": wandb.Image(img_constrastive),
        })
        return np.mean(np.asarray(log_record['loss/valid']))


    def draw_h(self, pair_data1, pair_data2, h_pair_data1, h_pair_data2):
        def get_solution(data):
            solution = data['p_to_p'].edge_index
            solution = solution[:, data['p_node'].batch[solution[0]] == 0]
            solution = solution[:, data['p_node'].path[solution[0],0]==data['p_node'].path[solution[1],0]]
            solution = data['p_node'].path[:, -1][solution]
            env_datas = data['env']
            ID_map, ID_graph, ID_CASE, num_agents = env_datas.ID_map[0], env_datas.ID_graph[0], env_datas.ID_CASE[0], env_datas.num_agents[0]
            ID_map, ID_graph, ID_CASE, num_agents = int(ID_map), int(ID_graph), int(ID_CASE), int(num_agents)                    
            pos = data['g_node'].pos[data['g_node'].batch==0].data.cpu().numpy()
            return solution, pos, ID_map, ID_graph, ID_CASE, num_agents, int(data.V[0])

        # plot pairing data
        solution1, pos, ID_map, ID_graph, ID_CASE, num_agents, V1 = get_solution(pair_data1)
        solution2, pos, ID_map, ID_graph, ID_CASE, num_agents, V2 = get_solution(pair_data2)                 

        h1, h2 = float(h_pair_data1[0]), float(h_pair_data2[0])

        img_constrastive = plot_v_and_h(self.Env_Simulator, ID_map, pos,
                     (solution1.data.cpu().numpy().T, solution2.data.cpu().numpy().T), (f'h: {h1}, V: {V1}', f'h: {h2}, V: {V2}'))

        return img_constrastive

            
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print(self.model)
        print("Experiment on {} finished.".format(self.config.exp_name))
        print("Please wait while finalizing the operation.. Thank you")
        # self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        self.data_loader.finalize()
