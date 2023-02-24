from copy import deepcopy
from time import time
from copy import deepcopy

from expert.RVOPlanner import RVOPlanner


from utils.dirs import return_name_prefix_Envconfig_NumAgent, return_dir_dataset_root
import numpy as np
# import gym.envs.classic_control.rendering as rendering
import time
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
	def plan(env, starts, goals, time_budget=float('inf'), animation=False, **kwargs):
		'''
		output:
			return an object if there is a result. The object should have object.makespan and object.flowtime and object.solution
		'''

		def normal_return():
			return solution

		try:
			solution = orca(env, starts, goals, timeout=time_budget,  animation=animation)
			if solution is None:
				return None
			return normal_return()

		except TimeOutException as e:
			print(e)
			solution = None
			return normal_return()
        
def orca(env, starts, goals, timeout,  animation=False,):
		'''
		output:
			return an object if there is a result. The object should have object.makespan and object.flowtime and object.solution
		'''

		planner = RVOPlanner()

		stepsize = 0.05
		planner.setup_scenario(env, starts, goals, stepsize=stepsize)

		viewer = None
		#------------------------------
		# simulation step
		done = False
		finish = False
		#------------------------------
		#simulation starts
		print("running ocra")
		t = 0
		start_t = time.perf_counter()
		while True:
			if animation:
				# if viewer is None:
				# 	viewer = rendering.Viewer(500, 500)
				# 	viewer.set_bounds(-1, 1, -1, 1)

				planner.update_visualization(viewer)
			planner.set_preferred_velocities()

			planner.simulator_.step()
			state = planner.get_states()
			#------------------------------
			finish, collision = env.step_state(state, t)
			done = planner.reached_goal()
			if done:
				print('all reachgoal', finish, collision, planner.simulator_.global_time_, env.agent_end_time)
				solution = Solution(env.agents_path, sum(env.agent_end_time), planner.simulator_.global_time)
				return solution
			if collision:
				return None
			if (time.perf_counter() -start_t)>timeout:
				raise TimeOutException("Timeout. Expert fails to find a solution.")
			t += 1

class Solution:
	def __init__(self, solution, flowtime, makespan):
		self.solution = solution
		self.flowtime = flowtime
		self.makespan = makespan