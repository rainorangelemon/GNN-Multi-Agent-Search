from copy import deepcopy
from time import time
from copy import deepcopy
from expert.RVO import RVO_update, reach, compute_V_des, reach
from expert.RVOVis import visualize_traj_dynamic
from utils.dirs import return_name_prefix_Envconfig_NumAgent, return_dir_dataset_root
import numpy as np

def pad_paths(paths):
	paths = deepcopy(paths)
	T = max([len(p) for p in paths])
	paths = [p + [p[-1] for t in range(T - len(p))] for p in paths]
	return paths


class TimeOutException(Exception):
	def __init__(self, message):
		super(TimeOutException, self).__init__(message)


class FailureException(Exception):
	def __init__(self, message):
		super(FailureException, self).__init__(message)


class ORCAPlanner:
	def __init__(self):
		pass

	@staticmethod
	def plan(env, starts, goals, env_setting, time_budget=float('inf'), animation=False):
		'''
		output:
			return an object if there is a result. The object should have object.makespan and object.flowtime and object.solution
		'''

		def normal_return():
			return solution

		try:
			solution = orca(env, starts, goals, env_setting, timeout=time_budget,  animation=animation)
			if solution is None:
				raise FailureException('cannot find a feasible solution')
			return normal_return()

		except TimeOutException as e:
			print(e)
			solution = None
			return normal_return()
        
def orca(env, starts, goals, env_setting, timeout,  animation=False,):
		'''
		output:
			return an object if there is a result. The object should have object.makespan and object.flowtime and object.solution
		'''

		# define workspace model
		ws_model = dict()
		#robot radius
		ws_model['robot_radius'] = env.RAD_DEFAULT
		#circular obstacles, format [x,y,rad]
		# no obstacles

		if env.config.env_name == 'BoxEnv':
			step = 0.001
			ws_model['circular_obstacles'] = []
			for box in env.obstacle_boxes:
				radius = np.sqrt(box['meta']['size'][0]**2 + box['meta']['size'][1]**2)/2
				ws_model['circular_obstacles'].append([box['shape'].centroid.x*env.scale, box['shape'].centroid.y*env.scale, radius*env.scale * 1.3])
		elif env.config.env_name == 'MazeEnv':
			step = 0.05# 0.002 #0.001
			ws_model['circular_obstacles'] = [[env._inverse_transform([i, j])[0]+(1 / env.width), env._inverse_transform([i, j])[1]+(1 / env.width), (1 / env.width)* 1.3] for j in range(env.map.shape[1]) for i in range(env.map.shape[0]) if env.map[i, j] == 1]

		
		env.reset(starts, goals)
		X = [pos for id_agent, pos in env.agents_cur_pos.items()]
		goal = [pos for id_agent, pos in env.agents_goal_pos.items()]

		V_max = [env.median_edge_weight for i in range(len(X))]
		V = [[0,0] for i in range(len(X))]

		#------------------------------
		# simulation step
		done = False
		finish = False
		#------------------------------
		#simulation starts
		t = 0
		while True:
			# compute desired vel to goal
			V_des, done = compute_V_des(X, goal, V_max, env.MIN_THRESHOLD)
			# compute the optimal vel to avoid collision
			V = RVO_update(X, V_des, V, ws_model)
			# update position
			for i in range(len(X)):
				norm_v = np.sqrt(V[i][0]**2 + V[i][1]**2)
				if norm_v > V_max[i]:
					V[i][0] = V[i][0] / norm_v * V_max[i]
					V[i][1] = V[i][1] / norm_v * V_max[i]
				V[i][0]=V[i][0]*step
				V[i][1]=V[i][1]*step
				X[i][0] += V[i][0]
				X[i][1] += V[i][1]
			#----------------------------------------
			# visualization
			finsih, collision = env.step(V, t)

			# finsih, collision = env.step(V, t)
			if done:
				print('all reachgoal', finish)
				solution = Solution(env.agents_path, env.agents_flowtime*step/env.median_edge_weight, env.agents_makespan*step/env.median_edge_weight)
				return solution
			# if collision:
			# 	return None
			if animation:
				data_path = return_dir_dataset_root(env.config)
				name_prefix = return_name_prefix_Envconfig_NumAgent(env.config.env_name, env_setting, len(X))
				path_to_expertSol = data_path/ 'result_OCRA' /f'{name_prefix}'
				path_to_expertSol.mkdir(parents=True, exist_ok=True)
				# print(path_to_expertSol)
				visualize_traj_dynamic(ws_model, X, V, goal, time=t*step, name=f'{path_to_expertSol}/snap{str(t/10)}.png')
			t += 1

class Solution:
	def __init__(self, solution, flowtime, makespan):
		self.solution = solution
		self.flowtime = flowtime
		self.makespan = makespan