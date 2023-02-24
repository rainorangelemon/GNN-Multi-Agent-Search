import os
from os.path import dirname, realpath, pardir
import argparse
from copy import deepcopy

config_base = {
  'env_name': 'MazeEnv',
  'num_map_source': 3000,

  'k': 6,
  'env_dim': 2,
  "n_nodes": 1000,
  "dim": 2,
  'num_agents': 5,

  "ID_Env_Start": 0,
  "ID_Env_End": 20,

  # "min_num_agents": 1,
  # "max_num_agents": 5,
  "min_max_num_agents": [2, 10],

  "only_constraint": False,
  'num_ProblemGraph_PerEnv': 20,
  'num_Cases_PerProblemGraph': 1,

  'split_train_end': 0.7,
  'split_valid_end': 0.85,

  "seed": 4123,
  'timeout': 300,

  'split_dataset': False,
  'shuffle_dataset': True,
  "shrink_split_dataset_end": 8000,
  "n_workers": 2,
  're_preprocess': False,
  'over_write_preprocess': False,
}

config_preprocess = {**config_base, **{
  'task': 'preprocess',
  'mode': 'all',
  'split_during_generate': True,
}}

config_experiment = {**config_base, **{
  'task': 'experiment',
  'mode': 'train',

  "exp_net": "AgentExplorerMultiAgentRegion",
  "exp_name": "AgentExplorerMultiAgentRegion",
  "exp_setting": "AgentExplorerMultiAgentRegion",

  "negative_sampling": "just_optimal",

  "cuda": True,
  "gpu_device":0,

  "learning_rate": 3e-4,
  "weight_decay": 0.00001,
  "momentum": 0.9,
  "max_epoch": 200000,
  "use_scheduler": False, 

  "lastest_epoch": True,
  "best_epoch": False,
  "test_epoch": 0,

  "log_interval": 10,
  "log_loss_interval": 40,
  "log_visual_interval": 200,
  "validate_every": 4,

  "data_loader_workers": 0,
  "pin_memory": True,
  "async_loading": False,
  
  "batch_size": 48,
  "valid_batch_size": 16,
  "test_batch_size": 1,

  "con_train": False,
  "train_TL": False,
  "log_time_trained": '0',

  # for GNN architecture
  "dim_te": 64,
  "dim_embed": 64,
  "temporal_encoding": True,
  "agent_identifier": True,    

  # for training
  "train_gamma": 0.1,

  # for inference
  "infer_w": float('inf'),  # the suboptimal bound
}}