import numpy as np
from utils.config import seed_everything
import torch
from torch_geometric.nn import knn_graph, radius_graph
from collections import defaultdict
from time import time
import pickle
from tqdm import tqdm
from torch_sparse import coalesce

INFINITY = float('inf')
# random_seed = 4123
# np.random.seed(random_seed)
# torch.random.manual_seed(random_seed)

def construct_graph(env, points, k=6, only_free_neighbor=False):
    edge_index = knn_graph(torch.FloatTensor(np.array(points)), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    edge_index = edge_index_torch.data.cpu().numpy().T
    edge_index_return = []
    edge_cost = defaultdict(list)
    edge_cost_max_free = 0
    edge_free = []
    neighbors = defaultdict(list)
    for i, edge in enumerate(edge_index):
        if env._edge_fp(points[edge[0]], points[edge[1]]):
            dist = np.linalg.norm(points[edge[1]] - points[edge[0]])
            edge_cost_max_free = max(edge_cost_max_free, dist)
            edge_cost[edge[1]].append(dist)
            edge_free.append(True)
            edge_index_return.append(edge)
            neighbors[edge[1]].append(edge[0])
        elif (not only_free_neighbor):
            edge_cost[edge[1]].append(INFINITY)
            edge_free.append(False)
            edge_index_return.append(edge)
            neighbors[edge[1]].append(edge[0])
    return edge_cost, neighbors, np.array(edge_index_return), edge_free, edge_cost_max_free


def construct_graph_radius(env, points, r, only_free_neighbor=False):
    edge_index = radius_graph(torch.FloatTensor(np.array(points)), r=r, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    edge_index = edge_index_torch.data.cpu().numpy().T
    edge_index_return = []
    edge_cost = defaultdict(list)
    edge_cost_max_free = 0
    edge_free = []
    neighbors = defaultdict(list)
    for i, edge in enumerate(edge_index):
        if env._edge_fp(points[edge[0]], points[edge[1]]):
            dist = np.linalg.norm(points[edge[1]] - points[edge[0]])
            edge_cost_max_free = max(edge_cost_max_free, dist)
            edge_cost[edge[1]].append(dist)
            edge_free.append(True)
            edge_index_return.append(edge)
            neighbors[edge[1]].append(edge[0])
        elif (not only_free_neighbor):
            edge_cost[edge[1]].append(INFINITY)
            edge_free.append(False)
            edge_index_return.append(edge)
            neighbors[edge[1]].append(edge[0])
    return edge_cost, neighbors, np.array(edge_index_return), edge_free, edge_cost_max_free


def construct_prm(env, data_path, random_seed=4123, k=6):

    data = []
    seed_everything(random_seed)

    time0 = time()
    results = []
    n_sample = 1000

    for problem_index in tqdm(range(2000)):

        env.init_new_problem(problem_index)
        points = []
        for _ in range(np.random.randint(100, 400)):
            points.append(env.uniform_sample())
        edge_cost, neighbors, edge_index, edge_free = construct_graph(env, points)

        data.append((points, neighbors, edge_cost, edge_index, edge_free))

    with open(data_path, 'wb') as f:
        pickle.dump(data, f, pickle.DEFAULT_PROTOCOL)
