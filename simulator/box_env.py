import os
import glob
import seaborn as sns
import numpy as np
import torch
import torchvision
import json
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from random import sample
from collections import defaultdict
from os.path import dirname, realpath, pardir
from pathlib import Path
from shapely.prepared import prep
from tqdm import tqdm

import shapely
from shapely.geometry import MultiPolygon, Polygon, LineString, Point
from shapely.ops import unary_union

from scipy.spatial.distance import cdist
from utils.construct_prm import construct_graph, construct_graph_radius
from matplotlib.collections import LineCollection

import shapely
from shapely.geometry import MultiPolygon, Polygon, LineString
from shapely.ops import unary_union
from utils.dirs import return_dir_datasource, return_name_prefix_Envconfig, return_name_prefix_Envconfig_NumAgent

from simulator.base_env import BaseEnv

## RAD_DEFAULT: for checking the inter-robot, robot-env collision
# RAD_DEFAULT = 0.35 / 2


def get_box_polygon(pos, angle, dim):
    box = shapely.geometry.box(-dim[0] / 2, -dim[1] / 2, dim[0] / 2, dim[1] / 2)
    box_rot = shapely.affinity.rotate(box, angle, use_radians=True)
    box_trans = shapely.affinity.translate(box_rot, pos[0], pos[1])
    return {
        "meta": {
            "t": [float(pos[0]), float(pos[1]), float(dim[2] / 2)],
            "r": [0.0, 0.0, 1.0, float(angle)],
            "size": [float(d) for d in dim],
        },
        "shape": box_trans,
    }


def restore_state(state):

    def restore_polygons(boxes_meta):
        return [
            get_box_polygon(meta["t"][:2], meta["r"][3], meta["size"])
            for meta in boxes_meta
        ]

    border_boxes = restore_polygons(state["border_obstacles"])
    obstacle_boxes = restore_polygons(state["inner_obstacles"])
    return border_boxes, obstacle_boxes


class BoxEnv(BaseEnv):
    '''
    Interface class for Box Environment
    '''

    def __init__(self, config):
        super().__init__(config)

        print(f"Initializing Box Environment")
        # self.config_dim = 2
        self.config = config

        # TODO: make this as an argument from hard coded value
        self.r = 0.125 #* 0.38 # Hz ~ 6-7
        self.k = config.k
        self.dim = config.dim
        self.n_nodes = config.n_nodes

        self.num_agents = config.num_agents
        self.map_file = Path(config.data_root)/f'{config.env_name}'/f"dataset_{config.num_map_source}"/'meta'
        self.data_path_source = return_dir_datasource(config)


    def __str__(self):
        return 'Box Environment'

    def load_map(self, index=None):
        with open(str(self.map_file/f'BoxEnv_{index:06}.json'), "r") as file:
            meta = json.load(file)

        self.meta = meta
        self.cfg = meta["cfg"]

        self.border_boxes, self.obstacle_boxes = restore_state(self.meta)

        self.occupied_area = shapely.ops.unary_union(
            [box["shape"] for box in list(self.obstacle_boxes)+list(self.border_boxes)]
        )
        # self.occupied_area_prep = prep(self.occupied_area)
        xmin, ymin, xmax, ymax = self.occupied_area.bounds
        self.scale = 1 / max((xmax-xmin)/2, (ymax-ymin)/2)
        # print('SCALE', self.scale)
        RAD_DEFAULT = 0.35 / 2
        self.RAD_DEFAULT = RAD_DEFAULT * self.scale
        self.MIN_THRESHOLD = 2 * self.RAD_DEFAULT * 1.1
        self.occupied_area = shapely.affinity.scale(self.occupied_area, xfact=self.scale, yfact=self.scale)
        self.occupied_area_prep = prep(self.occupied_area)
        self.xmin, self.ymin, self.xmax, self.ymax = self.occupied_area.bounds

    def get_problem(self):
        problem = {
            "meta": self.meta,
            "graph": self.graphs,
        }
        return problem


