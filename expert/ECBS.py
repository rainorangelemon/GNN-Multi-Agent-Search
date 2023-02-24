from expert.Astar_epsilon import *
from queue import PriorityQueue
from copy import deepcopy
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from expert.BFS import BFSPlanner
from time import time


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


class ECBSPlanner:
	def __init__(self):
		self.explored_nodes = []

	def plan(self, env, starts, goals, need_update_V=False,
			 time_budget=float('inf'), w=1.1, **kwargs):
		'''
		output:
			if need_update_V is True, then return optimal path and explored nodes
			else return optimal path
			optimal path is None if no solution is found
		'''

		self.explored_nodes = []
		optimal_solution = None

		def normal_return():
			if need_update_V:
				update_V(env, self.explored_nodes)
				return optimal_solution, self.explored_nodes
			else:
				return optimal_solution

		try:
			optimal_solution = ecbs(env, starts, goals, self.explored_nodes, timeout=time_budget, w=w)
			if optimal_solution is None:
				raise FailureException('cannot find a feasible solution')
			return normal_return()

		except TimeOutException as e:
			print(e)
			optimal_solution = None
			return normal_return()


def ecbs(env, starts, goals, explored_nodes, timeout, w=1.1):

	start_t = time()

	open_set = list()
	focal_set = list()

	backplanner = BFSPlanner(env, goals)
	root = Constraint(w, num_agents=len(starts))
	root.find_paths(env, starts, goals, backplanner)

	open_set.append(root)
	focal_set.append(root)

	bestCost = root.LB

	explored_nodes.append(root)
	while len(open_set):
		if (time()-start_t)>timeout:
			raise TimeOutException("Timeout. Expert fails to find a solution.")

		# root_cost, root_depth, root		
		best_node = min(open_set, key=lambda node: (node.LB))

		oldBestCost = bestCost
		bestCost = best_node.LB
		if bestCost > oldBestCost:
			for node in open_set:
				cost = node.cost
				if (cost > oldBestCost * w) and (cost <= bestCost * w):
					focal_set.append(node)

		cur_node = min(focal_set, key=lambda node: (node.focalHeuristic, node.cost))
		
		focal_set.remove(cur_node)
		open_set.remove(cur_node)

		conflict = find_conflict(env, cur_node.solution)

		if conflict is None:
			if (cur_node.cost != float('inf')):
				cur_node.V = 0
				update_V(env, explored_nodes)
				cur_node.solution = pad_paths(cur_node.solution)
				return cur_node
			else:
				print('ECBS failed!')
				return None
			
		else:

			def add_child(child):
				child.find_paths(env, starts, goals, backplanner)
				child_cost = child.cost
				open_set.append(child)
				if child_cost <= bestCost * w:
					focal_set.append(child)
				explored_nodes.append(child)

			if conflict['conflict_type'] == 'collided_obstacle':
				child = cur_node.create_child()
				is_new_constraint = child.add_constraint(agent=conflict['agent'], t=conflict['t'], node=conflict['node'])
				# TODO: the behaviour is strange if the following line is uncommented
				# assert is_new_constraint
				add_child(child)
			
			else:
				for agent in ['agent1', 'agent2']:
					child = best_node.create_child()
					if conflict['conflict_type'] == 'inter_robot_collision':
						is_new_constraint = child.add_constraint(agent=conflict[agent], t=conflict['t'], node=conflict[f'node_{agent}'])
					elif conflict['conflict_type'] == 'edge_collision':
						is_new_constraint = child.add_transition_constraint(agent=conflict[agent], t=conflict['t'], node=conflict[f'node_{agent}'], lastnode=conflict[f'node_{agent}_prev'])
					else:
						assert False
					# TODO: the behaviour is strange if the following line is uncommented
					# assert is_new_constraint
					add_child(child)

	return None


def update_V(env, explored_nodes):
	for node in explored_nodes[::-1]:
		update_V_recursive(env, node)


def update_V_recursive(env, node):
	if node.V is None:
		if len(node.children)==0:
			node.V = update_V_leaf(env, node)
		else:
			node.V = min([update_V_recursive(env, child) for child in node.children])
	return node.V


def update_V_leaf(env, node):
	return 1


class Constraint:

	def __init__(self, w, num_agents):
		self.new_agent_constraint = None
		self.constraints = {}
		self.transition_constraints = {}

		self.children = []
		self.solution = None
		self.cost = None
		self.num_constraints_total = 0
		# suboptimal bound w
		self.w = w
		self.V = None
		self.LB = 0
		self.num_agents = num_agents
		self.list_fScore = [0] * num_agents
		self.list_cost = [0] * num_agents
		self.focalHeuristic = 0
		self.list_focalHeuristic = np.zeros(self.num_agents)

	def create_child(self):
		child = Constraint(self.w, self.num_agents)
		child.new_agent_constraint = None
		child.constraints = deepcopy(self.constraints)
		child.transition_constraints = deepcopy(self.transition_constraints)
		child.solution = deepcopy(self.solution)
		child.list_focalHeuristic= deepcopy(self.list_focalHeuristic)
		child.list_cost = deepcopy(self.list_cost)
		child.list_fScore = deepcopy(self.list_fScore)
		child.focalHeuristic = deepcopy(self.focalHeuristic)
		child.LB = deepcopy(self.LB)
		child.num_constraints_total = self.num_constraints_total
		self.children.append(child)
		return child

	def add_constraint(self, agent, t, node):
		self.new_agent_constraint = agent
		self.num_constraints_total += 1
		# check one agent overlap another agent
		if agent not in self.constraints:
			self.constraints[agent] = {}
		if t not in self.constraints[agent]:
			self.constraints[agent][t] = set()
		if node in self.constraints[agent][t]:
			return False
		self.constraints[agent][t].add(node)
		return True

	def add_transition_constraint(self, agent, t, node, lastnode):
		self.new_agent_constraint = agent
		self.num_constraints_total += 1
		# check position swap between two agents in the same time
		if agent not in self.transition_constraints:
			self.transition_constraints[agent] = {}
		if t not in self.transition_constraints[agent]:
			self.transition_constraints[agent][t] = {}
		if node not in self.transition_constraints[agent][t]:
			self.transition_constraints[agent][t][node] = set()
		if lastnode in self.transition_constraints[agent][t][node]:
			return False
		self.transition_constraints[agent][t][node].add(lastnode)
		return True

	def get_constraint_fn(self, agent):
		def constraint_fn(node, lastnode, t):
			overlap = node in self.constraints.get(agent, {}).get(t, set())
			on_edge = lastnode in self.transition_constraints.get(agent, {}).get(t, {}).get(node, set())
			return (not overlap) and (not on_edge)
		return constraint_fn

	def find_paths(self, env, starts, goals, backplanner):

		paths = [None] * len(starts)
		for agent in range(len(starts)):

			if (self.new_agent_constraint is not None) and (self.new_agent_constraint!=agent):
				# if new contraint is not added, don't need to change its path
				path = self.solution[agent]
				paths[agent] = path
				
			else:
				T = 0
				if (agent in self.constraints) and len(self.constraints[agent]):
					T = max(T, max(self.constraints[agent].keys())+1)
				if (agent in self.transition_constraints) and (len(self.transition_constraints[agent])):
					T = max(T, max(self.transition_constraints[agent].keys())+1)
				path, cost, fScore, new_focal_h = astar_epsilon(env=env, start=starts[agent], goal=goals[agent], agent_id=agent,
																backplanner=backplanner, constraint_fn=self.get_constraint_fn(agent), 
																min_time=T, return_cost=True, paths=self.solution, w=self.w)
				self.list_focalHeuristic[agent] = new_focal_h
				self.list_cost[agent] = cost + 1  # we count the total time steps
				self.list_fScore[agent] = fScore + 1
				paths[agent] = path

		self.LB = sum(self.list_fScore)
		self.solution = paths
		self.makespan = max([len(p) for p in paths])
		self.flowtime = self.cost = sum([len(p) for p in paths])
		assert self.flowtime == sum(self.list_cost) 
		if self.new_agent_constraint == None:
			self.list_focalHeuristic = update_conflict_matrix(env, paths, self.list_focalHeuristic)
		self.focalHeuristic = np.sum(self.list_focalHeuristic)


def find_conflict(env, paths):

	paths = pad_paths(paths)
	num_agents = len(paths)
	
	for t in range(len(paths[0])):
		# Check collisions (agents on the same node, or inter agent collision)

		for agent_i in range(num_agents):
			v_i_t = paths[agent_i][t]
			v_i_t_pos = env.find_pos_of_node_from_graph(agent_i, v_i_t)

			for agent_j in range(agent_i+1, num_agents):
				v_j_t = paths[agent_j][t]
				v_j_t_pos = env.find_pos_of_node_from_graph(agent_j, v_j_t)

				if not env._state_fp(v_i_t_pos):
					# print(f'Agent {agent_i} collided_obstacle at {t}')
					return {'agent': agent_i, 't': t, 'node': v_i_t,
							'conflict_type': 'collided_obstacle'}

				if not env._state_fp(v_j_t_pos):
					# print(f'Agent {agent_j} collided_obstacle at {t}')
					return {'agent': agent_j, 't': t, 'node': v_j_t,
							'conflict_type': 'collided_obstacle'}

				if env.collide_static_agents(v_i_t_pos, v_j_t_pos):
					# print(f'Agent {agent_i}-{agent_j} inter_robot_collision at {t}')

					return {'t': t,
							'agent1': agent_i, 'node_agent1': v_i_t,
							'agent2': agent_j, 'node_agent2': v_j_t,
							'conflict_type': 'inter_robot_collision'}

				if t > 0:
					v_i_t_prev = paths[agent_i][t - 1]
					v_i_t_pos_prev = env.find_pos_of_node_from_graph(agent_i, v_i_t_prev)
					v_j_t_prev = paths[agent_j][t - 1]
					v_j_t_pos_prev = env.find_pos_of_node_from_graph(agent_j, v_j_t_prev)

					if env.continuous_collide_spheres(v_i_t_pos_prev, v_i_t_pos, v_j_t_pos_prev, v_j_t_pos):#, 0.05, 0.05):
						# print(f'Edge_collision at {t}:\n Agent {agent_i} \t({v_i_t_prev}->{v_i_t})\n Agent {agent_j} \t({v_j_t_prev}->{v_j_t}) ')
						return {'t': t,
								'agent1': agent_i, 'node_agent1': v_i_t, 'node_agent1_prev': v_i_t_prev,
								'agent2': agent_j, 'node_agent2': v_j_t, 'node_agent2_prev': v_j_t_prev,
								'conflict_type': 'edge_collision'}

	return None


def update_conflict_matrix(env, paths, list_focalHeuristic):
	paths = pad_paths(paths)
	num_agents = len(paths)
	
	# Check collisions (agents on the same node, or inter agent collision)

	for agent_i in range(num_agents):

		count_inter_collision = 0
		for agent_j in range(num_agents):
			if agent_i == agent_j:
				continue

			for t in range(len(paths[0])):
				v_i_t = paths[agent_i][t]
				v_i_t_pos = env.find_pos_of_node_from_graph(agent_i, v_i_t)
				v_j_t = paths[agent_j][t]
				v_j_t_pos = env.find_pos_of_node_from_graph(agent_j, v_j_t)

				count_inter_collision += (not env._state_fp(v_i_t_pos)) + (not env._state_fp(v_j_t_pos))

				count_inter_collision += env.collide_static_agents(v_i_t_pos, v_j_t_pos)

				if t > 0:
					v_i_t_prev = paths[agent_i][t - 1]
					v_i_t_pos_prev = env.find_pos_of_node_from_graph(agent_i, v_i_t_prev)
					v_j_t_prev = paths[agent_j][t - 1]
					v_j_t_pos_prev = env.find_pos_of_node_from_graph(agent_j, v_j_t_prev)

					count_inter_collision += env.continuous_collide_spheres(v_i_t_pos_prev, v_i_t_pos, v_j_t_pos_prev, v_j_t_pos)

		list_focalHeuristic[agent_i] = count_inter_collision

	return list_focalHeuristic

