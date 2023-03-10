"""
Example file showing a demo with 100 agents split in four groups initially positioned in four corners of the environment. Each agent attempts to move to other side of the environment through a narrow passage generated by four obstacles. There is no roadmap to guide the agents around the obstacles.
"""
import math
import random
import gym.envs.classic_control.rendering as rendering

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))

parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import expert.rvo.math as rvo_math

from expert.rvo.vector import Vector2
from expert.rvo.simulator import Simulator
from expert.RVOPlanner import *

RVO_RENDER = True




def main():
    from simulator.maze_env import MazeEnv
    from pathlib import Path
    from easydict import EasyDict

    Root_Path = '/media/qingbiao/Data/ql295/Data/MultiAgentSearch/'
    dir_root_save = Path(Root_Path)
    dir_dataset_map_save = dir_root_save/'dataset'/"MazeEnv"

    # ID_map, ID_graph, ID_CASE, numAgent = 299, 0, 0, 4
    # ID_map, ID_graph, ID_CASE, numAgent = 2012, 0, 0, 4
    ID_map, ID_graph, ID_CASE, numAgent = 2003, 0, 0, 7
    env_setting = (ID_map, ID_graph, ID_CASE)
    config_BoxEnv = {'env_name': 'MazeEnv',
                        
                        "exp_name": "AgentExplorerMultiAgent",
    #                       'data_root': '/home/qingbiao/PycharmProjects/GNN_MultiAgentSearch/dataset/', 
    #                       "save_animation": "/home/qingbiao/PycharmProjects/GNN_MultiAgentSearch/results/animation/",
                        
                        'data_root': '/media/qingbiao/Data/ql295/Data/MultiAgentSearch/dataset/', 
                        "save_animation": "/media/qingbiao/Data/ql295/Data/MultiAgentSearch/",
                        'num_map_source': 3000,   

                        'k': 6,
                        'env_dim': 2,
                        "n_nodes": 1000,
                        "dim": 2,
                        'num_agents': numAgent,
                        "config_dim": 2,

                        "num_Env": 20,

                        "min_num_agents": 2,
                        "max_num_agents": 10,


                        'num_ProblemGraph_PerEnv': 5,
                        'num_Cases_PerProblemGraph': 1,

                        'split_train_end': 0.7,
                        'split_valid_end': 0.85,

                        "seed": 4123,
                        'timeout': 300,
                        }
    config_setup = EasyDict(config_BoxEnv)

    dir_dataset_sol_dir = dir_dataset_map_save/f"dataset_3000_minNumAgent_{config_setup.min_num_agents}_maxNumAgent_{config_setup.max_num_agents}"/"Source"

    path_file_graph = dir_dataset_sol_dir/f'MazeEnv_ID_map_{ID_map:03d}_ID_graph_{ID_graph:03d}_ID_case_{ID_CASE:03d}_numAgent_{numAgent:03d}.graph'

    import pickle


    with open(path_file_graph, "rb") as file:
        data_dict_graph = pickle.load(file)

    viewer = None

    blocks = RVOPlanner()

    env = MazeEnv(config_setup)
    env.load_map(index=ID_map)
    env.init_new_problem_graph(index=ID_map, graphs=data_dict_graph['graphs'])
    stepsize=0.05
    # Set up the scenario.
    blocks.setup_scenario(env, data_dict_graph['init_states'], data_dict_graph['goal_states'], stepsize=stepsize)

    RVO_RENDER = True
    viewer = None
    print("Start simulation")
    # Perform (and manipulate) the simulation.
    # while not blocks.reached_goal():
    t = 0
    while True:
        if RVO_RENDER:
            if viewer is None:
                viewer = rendering.Viewer(500, 500)
                viewer.set_bounds(-1, 1, -1, 1)

            blocks.update_visualization(viewer)
        blocks.set_preferred_velocities()
        # print("vel", vel)
        blocks.simulator_.step()
        state = blocks.get_states()
        # print(state)
        finish, collsion = env.step_state(state, t)

        # env.render()
        # breakpoint()
        # list_obs = []
        # for obstacle in blocks.obstacles_:
        #     list_obs.append(obstacle)
        # blocks.simulator_.agents_[0].position_ = Vector2(0.0, 0.0)
        # blocks.simulator_.agents_[1].position_ = Vector2(0.2, 0.2)
        # blocks.simulator_.agents_[2].position_ = Vector2(1.0, 1.0)
        # blocks.simulator_.agents_[3].position_ = Vector2(-0.2, -0.2)
        # blocks.simulator_.agents_[4].position_ = Vector2(-0.8, -0.8)
        # for i in range(env.num_agents):
        #     print(f"agent {i} pos {blocks.simulator_.agents_[i].position_} goal {blocks.goals_[i]}")
        #     print(f"env agent {i} pos {env.agents_cur_pos[i]} goal {blocks.goals_[i]}")
        # breakpoint()
        t += 1
        if blocks.reached_goal():
            print("Goal reached", finish, env.agent_reached_goal, blocks.simulator_.global_time_, env.agent_end_time)
            solution = Solution(env.agents_path, sum(env.agent_end_time), blocks.simulator_.global_time_)

            break

class Solution:
	def __init__(self, solution, flowtime, makespan):
		self.solution = solution
		self.flowtime = flowtime
		self.makespan = makespan
if __name__ == '__main__':
    main()