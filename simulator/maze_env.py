import os
import glob
import seaborn as sns
import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from random import sample
from collections import defaultdict
from os.path import dirname, realpath, pardir
from matplotlib.lines import Line2D
from shapely.prepared import prep
from matplotlib.collections import LineCollection

import shapely
from shapely.geometry import MultiPolygon, Polygon, LineString, Point
from shapely.ops import unary_union

from scipy.spatial.distance import cdist
from utils.construct_prm import construct_graph
from utils.config import DotDict
from utils.dirs import return_dir_datasource, return_name_prefix_Envconfig, return_name_prefix_Envconfig_NumAgent, return_dir_dataset_root
from simulator.base_env import BaseEnv


RRT_EPS = 1e-2
STICK_LENGTH = 1.5 * 2 / 15
LIMITS = np.array([1., 1., 8.*RRT_EPS])

# MIN_THRESHOLD: to check whether agent reach goal
# MIN_THRESHOLD = 4.1e-2 #0.01#
RAD_DEFAULT = 2e-2


def get_box_polygon(pos, angle, dim):
    box = shapely.geometry.box(-dim[0] / 2, -dim[1] / 2, dim[0] / 2, dim[1] / 2)
    box_rot = shapely.affinity.rotate(box, angle, use_radians=True)
    box_trans = shapely.affinity.translate(box_rot, pos[0], pos[1])
    return {
        "shape": box_trans,
    }


class MazeEnv(BaseEnv):
    '''
    Interface class for maze environment
    '''

    RRT_EPS = RRT_EPS

    def __init__(self, config):
        print("Initializing Maze environment...")
        super().__init__(config)

        self.k = config.k
        self.dim = config.dim
        self.config = config
        
        self.n_nodes = config.n_nodes
        self.random_seed = config.seed
        self.num_agents = config.num_agents

        # load map from file
        dir_data_save = Path(config.data_root)/f'{config.env_name}'/"MazeEnvSource"
        self.data_path_source = return_dir_datasource(config)

        name_dataset_save = str(dir_data_save/'mazes_15_2_3000.npz')
        with np.load(name_dataset_save) as f:
            self.maps = f['maps']
            self.init_states = f['init_states']
            self.goal_states = f['goal_states']

        self.size = self.maps.shape[0]
        self.width = self.maps.shape[1]
        self.order = list(range(self.size))


    def __str__(self):
        return 'Maze Env'

    def load_map(self, index=None):

        self.map = self.maps[self.order[index]]
        self.occupied_area = shapely.ops.unary_union(
            [box["shape"] for box in [get_box_polygon(self._inverse_transform([i, j]), 0, (2 / self.width, 2 / self.width)) for j in range(self.map.shape[1]) for i in range(self.map.shape[0]) if self.map[i, j] == 1]]
        )
        self.occupied_area_prep = prep(self.occupied_area)
        # print(f'{self.config.env_name}:{self.occupied_area}')

        self.MIN_THRESHOLD = 4.1e-2  # 0.01#
        self.RAD_DEFAULT = 2e-2
        self.xmin, self.ymin, self.xmax, self.ymax = self.occupied_area.bounds

    def get_problem(self):
        #TOASK: Where to use it?
        problem = {
            "map": self.map,
            "graph": self.graphs,
        }
        return problem

