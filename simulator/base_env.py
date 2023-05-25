"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
from collections import defaultdict
from email.policy import default
import logging
import glob
from turtle import color
import numpy as np
from utils.dirs import return_dir_datasource, return_name_prefix_Envconfig, return_name_prefix_Envconfig_NumAgent
from random import sample
from utils.HaltonSampler import halton_samplers
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import PatchCollection, EllipseCollection, LineCollection
import seaborn as sns
import shapely
from shapely.geometry import MultiPolygon, Polygon, LineString, Point
from shapely.ops import unary_union
from pathlib import Path
from scipy.spatial.distance import cdist
from matplotlib import colors
from utils.construct_prm import construct_graph, construct_graph_radius
from abc import ABC, abstractmethod
from utils.config import DotDict
from copy import deepcopy
import matplotlib.colors as mc
import colorsys


class BaseEnv(ABC):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.dim = None
        self.ymax = None
        self.xmax = None
        self.occupied_area = None
        self.MIN_THRESHOLD = None
        self.ymin = None
        self.xmin = None
        self.graphs = None
        self.config = config
        self.num_agents = config.num_agents

        # self.k = config.k
        # self.dim = config.dim
        # self.config_dim = config.config_dim
        self.logger = logging.getLogger("Agent")

        construct_graphs_lambda = {
            'BoxEnv': lambda env, points, only_free_neighbor: construct_graph_radius(env, points, env.r, only_free_neighbor),
            'MazeEnv': construct_graph,
        }
        self.construct_graph = construct_graphs_lambda[self.config.env_name]

        # self.data_path_source = return_dir_datasource(config)


    @abstractmethod
    def load_map(self, index=None):
        pass

    def reset(self, agent_start_idx, agent_goal_idx):
        self.num_agents = len(agent_start_idx)
        self.agents_cur_pos = {} #np.zeros((self.num_agents, 2))
        self.agents_cur_state = np.zeros((self.num_agents, 2))
        self.agents_goal_pos = {} #np.zeros((self.num_agents, 2))
        self.agents_path = defaultdict(list)
        self.agents_flowtime = 0
        self.agents_makespan = 0
        self.agent_end_time = np.zeros(self.num_agents)
        self.agent_reached_goal = [False] * self.num_agents
        self.agent_check_collision = [False] * self.num_agents

        for id_agent in range(self.num_agents):
            self.agents_cur_pos[id_agent] = self.graphs['points'][agent_start_idx[id_agent]] 
            self.agents_cur_state[id_agent] = self.graphs['points'][agent_start_idx[id_agent]] 
            self.agents_path[id_agent].append(self.graphs['points'][agent_start_idx[id_agent]])
            self.agents_goal_pos[id_agent] = self.graphs['points'][agent_goal_idx[id_agent]] 

    def step(self, v, currentstep):
        inter_agent_check_collision = False

        for id_agent in range(self.num_agents):
            
            self.agents_cur_pos[id_agent] += v[id_agent]
            self.agents_cur_state[id_agent] = self.agents_cur_pos[id_agent]
            self.agents_path[id_agent].append(deepcopy(self.agents_cur_pos[id_agent]))
            if self.in_goal_region_single_pos(self.agents_cur_pos[id_agent], self.agents_goal_pos[id_agent]):
                self.agent_reached_goal[id_agent] = True
                self.agent_end_time[id_agent] = currentstep
            if not self._state_fp(self.agents_cur_pos[id_agent]):
                print('agent {} collided'.format(id_agent))
                self.agent_check_collision[id_agent] = True
        
        dist_inter_agent = cdist(self.agents_cur_state, self.agents_cur_state, metric='euclidean')
        np.fill_diagonal(dist_inter_agent, np.inf)
        if dist_inter_agent.min() < self.MIN_THRESHOLD:
            print('inter agent collision')
            inter_agent_check_collision = True
        collision  = any(self.agent_check_collision) or inter_agent_check_collision

        done = all(self.agent_reached_goal)

        if done:
            self.agents_flowtime = sum(self.agent_end_time)
            self.agents_makespan = max(self.agent_end_time)

        return done, collision

    def step_state(self, states, currenttime):
        inter_agent_check_collision = False

        for id_agent in range(self.num_agents):
            self.agents_cur_pos[id_agent] = np.array(states[id_agent])
            self.agents_path[id_agent].append(deepcopy(self.agents_cur_pos[id_agent]))
            if self.in_goal_region_single_pos(self.agents_cur_pos[id_agent], self.agents_goal_pos[id_agent]):
                self.agent_reached_goal[id_agent] = True
                if self.agent_end_time[id_agent] == 0.0:
                    self.agent_end_time[id_agent] = currenttime
            if not self._state_fp(self.agents_cur_pos[id_agent]):
                print('agent {} collided'.format(id_agent))
                self.agent_check_collision[id_agent] = True
        
        dist_inter_agent = cdist(self.agents_cur_state, self.agents_cur_state, metric='euclidean')
        np.fill_diagonal(dist_inter_agent, np.inf)
        if dist_inter_agent.min() < self.MIN_THRESHOLD:
            print('inter agent collision')
            inter_agent_check_collision = True
        collision  = any(self.agent_check_collision) or inter_agent_check_collision

        done = all(self.agent_reached_goal)

        if done:
            self.agents_flowtime = sum(self.agent_end_time)
            self.agents_makespan = max(self.agent_end_time)

        return done, collision

    def init_new_problem_graph(self, index, graphs=None):

        self.load_map(index)
        self.graphs = graphs

        if self.graphs is None:
            self.graphs = self.sample_graphs()
        
        list_edge_cost = []
        for key, value in self.graphs['edge_cost'].items():
            list_edge_cost.extend(value)
        self.median_edge_weight = np.median(list_edge_cost)
        self.avg_edge_weight = np.mean(list_edge_cost)
        self.min_edge_weight = np.min(list_edge_cost)

    def sample_graphs(self):
        points = self.sample_n_points(self.n_nodes, no_overlap=False)
        edge_cost, neighbors, edge_index, _, _ = self.construct_graph(self, points, only_free_neighbor=True)
        
        self.graph = DotDict({'points': points,
                                'edge_cost': edge_cost,
                                'neighbors': neighbors,  # list of neighbors for each node
                                'edge_index': edge_index})

        return self.graph

    def init_new_problem_instance(self, index=None):
        '''
        Initialize a new planning problem
        '''
        self.agent_starts = []
        self.agent_starts_vidxs = []
        self.agent_goals = []
        self.agent_goals_vidxs = []
        self.list_node_idx = list(range(self.n_nodes))
        for id_agent in range(self.num_agents):
            if id_agent > 0:
                while True:
                    start, goal = self.sample_start_goal(self.list_node_idx)
                    start_pos = self.graphs['points'][start]
                    goal_pos = self.graphs['points'][goal]
                    if (cdist([start_pos], self.agent_starts).min() > 2*self.RAD_DEFAULT) and (cdist([start_pos], self.agent_goals).min() > 2*self.RAD_DEFAULT) \
                            and (cdist([goal_pos], self.agent_starts).min() > 2*self.RAD_DEFAULT)  and (cdist([goal_pos], self.agent_goals).min() > 2*self.RAD_DEFAULT):
                        self.agent_starts.append(start_pos)
                        self.agent_starts_vidxs.append(start)
                        self.agent_goals.append(goal_pos)
                        self.agent_goals_vidxs.append(goal)
                        break
            else:
                start, goal = self.sample_start_goal(self.list_node_idx)
                start_pos = self.graphs['points'][start]
                goal_pos = self.graphs['points'][goal]
                self.agent_starts.append(start_pos)
                self.agent_starts_vidxs.append(start)
                self.agent_goals.append(goal_pos)
                self.agent_goals_vidxs.append(goal)

        return self.get_problem()

    def sample_start_goal(self, list_node_idx):
        start, goal = sample(list_node_idx, 2)
        return start, goal   

    def sample_n_points(self, n, no_overlap=False, need_negative=False):
        # TODO: distance to the existing nodes should be over a threshold
        if need_negative:
            negative = []
        samples = []
        candidates = []
        while len(samples) < n:
            if len(candidates):
                sample = candidates.pop().reshape(-1)
            else:
                candidates = list(self.uniform_sample(n=n))
                sample = candidates.pop().reshape(-1)
            if self._point_in_free_space(sample) and ((not no_overlap) or len(samples) == 0 or cdist([sample], samples).min() > (2*self.RAD_DEFAULT)):
                samples.append(sample)
            elif need_negative:
                negative.append(sample)
        if not need_negative:
            return samples
        else:
            return samples, negative

    def get_problem(self):
        raise NotImplementedError
    
    def uniform_sample(self, n=1):
        '''
        Uniformlly sample in the configuration space
        '''
        sample = halton_samplers[self.dim].random(n=n)
        sample = sample * np.array([self.xmax-self.xmin, self.ymax-self.ymin]).reshape(1, -1) + np.array([self.xmin, self.ymin]).reshape(1, -1)
        if n==1:
            return sample.reshape(-1)
        else:
            return sample

    def distance(self, agent_id, from_state, to_state):
        '''
        Distance metric: euclidean distance
        '''
        from_state = np.array(self.find_pos_of_node_from_graph(agent_id, from_state))

        to_state = np.array(self.find_pos_of_node_from_graph(agent_id, to_state))
        diff = np.abs(to_state - from_state)
        if diff.ndim == 1:
            diff = diff.reshape(1, -1)

        return np.sqrt(np.sum(diff ** 2, axis=-1))#/self.graphs[agent_id].edge_cost_max_free

    def in_goal_region_single(self, agent_id, state, goal):
        '''
        Return whether a single state(configuration) is in the goal region
        '''
  
        state = np.array(self.find_pos_of_node_from_graph(agent_id, state))
        goal = np.array(self.find_pos_of_node_from_graph(agent_id, goal))
        return (np.linalg.norm(goal-state)<self.MIN_THRESHOLD) and self._state_fp(state)
     
    def in_goal_region_single_pos(self, state, goal):
        '''
        Return whether a single state(configuration) is in the goal region
        '''
        return (np.linalg.norm(goal-state)<self.MIN_THRESHOLD) and self._state_fp(state)

    def find_pos_of_node_from_graph(self, agent_id, node_id):
        return self.graphs['points'][node_id]

    def get_potential_state_edgecost(self, agent_id, curr_node):
        # return neighbor of the agent's current state as candidate actions

        candidates = self.graphs['neighbors'][curr_node]
        candidates_edge_cost = self.graphs['edge_cost'][curr_node]

        return zip(candidates, candidates_edge_cost)

    def render(self, save_name=None):
        plt.clf()
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        if self.config.env_name == 'MazeEnv':
            self.plot_polygon(self.occupied_area, ax=ax, alpha=1.0, fc='#253494', ec='#253494')
            
            # for agent in self.agent_states:
            #     circle = patches.Circle((agent[0], agent[1]),  radius=self.RAD_DEFAULT, edgecolor='red', facecolor='red')
            #     plt.gca().add_patch(circle)

            for curr_node in self.graphs['points']:
                pos_x, pos_y = curr_node
                circle = patches.Circle((pos_x, pos_y),  radius=0.8*self.RAD_DEFAULT, edgecolor='lightgrey', facecolor='lightgrey')
                plt.gca().add_patch(circle)

            src_poses = []
            end_poses = []
            for edge in self.graphs['edge_index']:
                src_pos = self.find_pos_of_node_from_graph(0, edge[0])
                end_pos = self.find_pos_of_node_from_graph(0, edge[1])
                src_poses.append(src_pos)
                end_poses.append(end_pos)
            pos_xy_pos = np.array(list(zip(src_poses, end_poses)))
            lines = LineCollection(np.array(pos_xy_pos), color='grey', linestyle='solid', 
                                   linewidth=2, zorder=0)
            plt.gca().add_collection(lines)                
                
        elif self.config.env_name == 'BoxEnv':
            self.plot_polygon(self.occupied_area, ax=ax, alpha=1.0, fc='saddlebrown', ec='none')
            
            for agent in self.graphs['points']:
                circle = patches.Circle((agent[0], agent[1]), radius=self.RAD_DEFAULT, edgecolor='grey', facecolor='snow')
                plt.gca().add_patch(circle)
                
            src_poses = []
            end_poses = []
            for edge in self.graphs['edge_index']:
                src_pos = self.find_pos_of_node_from_graph(0, edge[0])
                end_pos = self.find_pos_of_node_from_graph(0, edge[1])
                src_poses.append(src_pos)
                end_poses.append(end_pos)
            pos_xy_pos = np.array(list(zip(src_poses, end_poses)))
            lines = LineCollection(np.array(pos_xy_pos), color='grey', linestyle='solid', 
                                   linewidth=2, zorder=0)
            plt.gca().add_collection(lines)

        fig = plt.gcf()
        plt.axis('off')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        if save_name is not None:
            fig.savefig(f'{save_name}_render.pdf', bbox_inches="tight")
        return

    @staticmethod
    def adjust_lightness(color, amount=0.5):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    def render_paths(self, paths, file_name):

        assert (('.png' in file_name) or ('.pdf' in file_name))
        plt.clf()
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        if self.config.env_name == 'MazeEnv':
            self.plot_polygon(self.occupied_area, ax=ax, alpha=1.0, fc='#253494', ec='#253494')
        elif self.config.env_name == 'BoxEnv':
            self.plot_polygon(self.occupied_area, ax=ax, alpha=1.0, fc='saddlebrown', ec='none')

        self.num_agents = len(paths)
        self.list_color = sns.color_palette("hls", self.num_agents)

        inheriter = [None]*len(paths)
        last_color = [None]*len(paths)
        makespan = len(paths[0])
        zorder = len(paths)+2
        for agent_id, path in enumerate(paths):
            len_path = len(path)
            if self.config.env_name=='MazeEnv':
                min_alpha = 0.4
            else:
                min_alpha = 0.2
            remap = (1-min_alpha)/len_path
             
            initial_z = zorder
            for id_path, curr_node in enumerate(path):
                
                previous_inheriter = deepcopy(inheriter[agent_id])
                previous_color = deepcopy(last_color[agent_id])
                if id_path < (len(path)-1):
                    if np.all(np.array(path[id_path])==np.array(path[id_path+1])):
                        inheriter[agent_id] = inheriter[agent_id] if (inheriter[agent_id] is not None) else id_path
                        last_color[agent_id] = last_color[agent_id] if (last_color[agent_id] is not None) else BaseEnv.adjust_lightness(self.list_color[agent_id], amount=remap*id_path+min_alpha)
                        continue
                    else:
                        inheriter[agent_id] = None
                        last_color[agent_id] = None
                label = f'{previous_inheriter}-{id_path}' if (previous_inheriter is not None) else str(id_path)                  
                
                pos_x, pos_y = self.find_pos_of_node_from_graph(agent_id, curr_node)
                color = BaseEnv.adjust_lightness(self.list_color[agent_id], amount=remap*id_path+min_alpha) if (previous_inheriter is None) else previous_color
                circle = patches.Circle((pos_x, pos_y), radius=0.8*self.RAD_DEFAULT if self.config.env_name=='MazeEnv' else self.RAD_DEFAULT,
                                        edgecolor=self.list_color[agent_id] if self.config.env_name=='BoxEnv' else color, facecolor=color, zorder=zorder) 
                
                if self.config.env_name=='MazeEnv' and (id_path<(len(path)-1)):
                    src_pos = self.find_pos_of_node_from_graph(agent_id, path[id_path])
                    end_pos = self.find_pos_of_node_from_graph(agent_id, path[id_path+1])
                    pos_xy_pos = list(zip([src_pos], [end_pos]))
                    lines = LineCollection(np.array(pos_xy_pos), color=color, linestyle='solid', 
                                           linewidth=2, zorder=initial_z)
                    ax.add_collection(lines)

                if self.config.env_name=='BoxEnv':
                    ax.text(pos_x, pos_y, label, fontsize=12, 
                            color=self.list_color[agent_id] if ((id_path if previous_inheriter is None else previous_inheriter)<=makespan//2) else 'white', 
                            zorder=zorder+1, ha='center', va='center')  #self.list_color[agent_id]
                else:
                    if id_path==0:
                        label = 's'
                    elif id_path==(len(path)-1):
                        label = 'g'
                    else:
                        label = None
                    if label is not None:
                        ax.text(pos_x, pos_y, label, fontsize=8, 
                                color=self.list_color[agent_id] if ((id_path if previous_inheriter is None else previous_inheriter)<=makespan//2) else 'white', 
                                zorder=zorder+1, ha='center', va='center')  #self.list_color[agent_id]
                zorder += 2
                plt.gca().add_patch(circle)

        fig = plt.gcf()
        plt.axis('off')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        fig.savefig(file_name, bbox_inches="tight")
        return

    def render_paths_animate(self, paths, goal_idxs, file_name, interpolate=20, dpi=100):
        assert (('.gif' in file_name) or ('.mp4' in file_name))
        
        RAD_DEFAULT = self.RAD_DEFAULT  #(0.8*self.RAD_DEFAULT if self.config.env_name=='MazeEnv' else self.RAD_DEFAULT)
        
        plt.clf()
        plt.close('all')

        if len(np.array(paths).shape) == 2:
            paths = [[self.find_pos_of_node_from_graph(agent_id, vid) for vid in p] for agent_id, p in enumerate(paths)]
        else:
            paths = paths[:, :, :]
        num_agents, makespan = len(paths), max([len(path) for path in paths])
        agents = np.transpose(np.array(paths), axes=(1, 0, 2))

        list_color = sns.color_palette("hls", num_agents)
        figsize = 40
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        ax = fig.add_subplot(111)

        rec_rad = 2*RAD_DEFAULT
        for agent_id, pos_idx, color in zip(range(num_agents), goal_idxs, list_color):
            pos = self.find_pos_of_node_from_graph(agent_id, pos_idx)
            rect = matplotlib.patches.Rectangle(pos-0.5*rec_rad, rec_rad, rec_rad, color=BaseEnv.adjust_lightness(color, amount=0.5))
            ax.add_patch(rect)

        if self.config.env_name == 'MazeEnv':
            self.plot_polygon(self.occupied_area, ax=ax, alpha=1.0, fc='#253494', ec='#253494')
        elif self.config.env_name == 'BoxEnv':
            self.plot_polygon(self.occupied_area, ax=ax, alpha=1.0, fc='saddlebrown', ec='none')
        ax.set_aspect('equal', adjustable='box')

        curr_agents = agents[0]
        num_agents = self.num_agents

        agent_circles = EllipseCollection([2*RAD_DEFAULT] * num_agents,
                                          [2*RAD_DEFAULT] * num_agents,
                                          np.zeros(num_agents),
                                          offsets=curr_agents, units='x',
                                          color=list_color,
                                          transOffset=ax.transData, )
        ax.add_collection(agent_circles)
        self.agent_circles = agent_circles

        self.texts = []
        for agent_id, pos in enumerate(curr_agents):
            pos_x, pos_y = pos
            text_current = ax.text(pos_x, pos_y, f'{agent_id}', fontsize=figsize*(12 if self.config.env_name=='MazeEnv' else 20)/10,
                    color='black', ha='center', va='center')
            self.texts.append(text_current)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_axis_off()
        plt.tight_layout()

        def update(frame_number):
            ratio = ((frame_number%interpolate)/interpolate)
            curr_agents = (1-ratio)*agents[frame_number//interpolate] + ratio*agents[min(1+frame_number//interpolate, makespan-1)]
            num_agents = self.num_agents
            self.agent_circles.remove()

            agent_circles = EllipseCollection([2*RAD_DEFAULT] * num_agents,
                                              [2*RAD_DEFAULT] * num_agents,
                                          np.zeros(num_agents),
                                          offsets=curr_agents, units='x',
                                          color=list_color,
                                          transOffset=ax.transData, )
            ax.add_collection(agent_circles)
            self.agent_circles = agent_circles

            for agent_id, pos in enumerate(curr_agents):
                self.texts[agent_id].set_position(pos)

        # Construct the animation, using the update function as the animation director.
        animation = FuncAnimation(fig, update, frames=makespan*interpolate, interval=1)
        if '.mp4' in file_name:
            writermp4 = FFMpegWriter(fps=interpolate)
            animation.save(file_name, writer=writermp4)
        if '.gif' in file_name:
            animation.save(file_name, writer=PillowWriter(fps=10))  # 'imagemagick', fps=10)

    def save_gif(self, env_setting, label_sol):
        name_prefix = return_name_prefix_Envconfig_NumAgent(self.config.env_name, env_setting, self.num_agents)

        dir_animation_case = Path(self.config.save_animation)/f"{self.config.exp_name}"/f"{self.config.env_name}"/f"{name_prefix}"
        dir_animation_gif = dir_animation_case /f"gif_{label_sol}"
        dir_animation_gif.mkdir(parents=True, exist_ok=True)

        dir_img_file = dir_animation_case / f'source_{label_sol}'
        name_gif_file = dir_animation_gif / f"{name_prefix}_{label_sol}.gif"

        print(env_setting, dir_img_file)
        print(env_setting, name_gif_file)
        image_list = []
        for filename in sorted(glob.glob(f'{dir_img_file}/*.png'), key=lambda x: (len(x), x)):  # assuming gif
            im = Image.open(filename)
            image_list.append(im)
        a = np.stack(image_list)

        image_list[0].save(name_gif_file, save_all=True, append_images=image_list[1:], loop=0, duration=50)
        return name_gif_file

    # =====================internal collision check module=======================
    def plot_polygon(self, polygon, ax, **kwargs):
        if isinstance(polygon, Polygon):
            polygon = MultiPolygon([polygon])
        assert isinstance(polygon, MultiPolygon)

        for p in polygon.geoms:
            ax.fill(*p.exterior.xy, **kwargs)
            for interior in p.interiors:
                ax.fill(*interior.xy, **kwargs)


    def _transform(self, state, w=15):
        coord = ((np.array(state)[:2].flatten() + 1.0) * w / 2.0).astype(int)
        coord[coord > w - 1] = w - 1
        return coord
    
    def _inverse_transform(self, coord, w=15):
        state = (np.array(coord) * 2.0 / w) - 1.0
        return state

    def _valid_state(self, state):
        return (self.xmax >= state[0] >= self.xmin) and \
               (self.ymax >= state[1] >= self.ymin)


    def _point_in_free_space(self, state):
        assert state.size == 2
        if not self._valid_state(state):
            return False
        point = Point(*state.reshape(-1)).buffer(self.RAD_DEFAULT)
        # point = Point(*state).buffer(self.RAD_DEFAULT)
        # print(state, self.occupied_area_prep.intersects(point))
        # breakpoint()
        return not self.occupied_area_prep.intersects(point)

    def _state_fp(self, state):
        # don't collide with the occupied_area
        return self._point_in_free_space(state)

    def _edge_fp(self, state, new_state):
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._state_fp(state) or not self._state_fp(new_state):
            return False

        return not self.occupied_area_prep.intersects(LineString([Point(*state.reshape(-1)), Point(*new_state.reshape(-1))]).buffer(self.RAD_DEFAULT))

    def collide_static_agents(self,
        pos1: np.ndarray,
        pos2: np.ndarray,
        size1 = None,
        size2 = None
    ) -> bool:
        """private func, detect collision between two static agents (sphere)

        Args:
            pos1 (np.ndarray): 'from' position of agent 1
            size1 (float): radius of agent 1
            pos2 (np.ndarray): 'from' position of agent 2
            size2 (float): radius of agent 2

        Returns:
            bool: true -> collide
        """
        # return collide_spheres(pos1, size1, pos2, size2)
        # fast check
        if size1 is None:
            size1 = self.RAD_DEFAULT
        if size2 is None:
            size2 = self.RAD_DEFAULT
        return Point(*pos1).buffer(size1).intersects(Point(*pos2).buffer(size2))

    def continuous_collide_spheres(self,
            from1: np.ndarray,
            to1: np.ndarray,
            from2: np.ndarray,
            to2: np.ndarray,
            rad1 = None,
            rad2 = None,
    ) -> bool:
        """detect collision between two dynamic spheres
        """
        if rad1 is None:
            rad1 = self.RAD_DEFAULT
        if rad2 is None:
            rad2 = self.RAD_DEFAULT

        return LineString([Point(*from1), Point(*to1)]).buffer(rad1).intersects(LineString([Point(*from2), Point(*to2)]).buffer(rad2))