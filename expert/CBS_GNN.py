from expert.Astar import *
from queue import PriorityQueue
from copy import deepcopy
from expert.BFS import BFSPlanner
from time import time
from copy import deepcopy
from utils.draw_tree import draw_tree
import heapq


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


class GNNPlanner:
	def __init__(self):
		self.explored_nodes = []

	def plan(self, env, starts, goals, model, w=float('inf'), need_update_V=False,
			 time_budget=float('inf'), plot_tree=None, **kwargs):
		'''
		output:
			if need_update_V is True, then return optimal path and explored nodes
			else return optimal path
			optimal path is None if no solution is found
		'''

		self.explored_nodes = []
		optimal_solution = None

		def normal_return():
			if (plot_tree is not None):
				print(f'---------------------{plot_tree}----------------------')
				draw_tree(self.explored_nodes)
			if need_update_V:
				update_V(env, self.explored_nodes)
				cbs_nodes = set()
				for node in self.explored_nodes:
					if node.V == 0:
						cbs_nodes.add(node)
						for child in node.children:
							cbs_nodes.add(child)
				return optimal_solution, list(cbs_nodes)
			else:
				return optimal_solution

		try:
			optimal_solution = focal_cbs(env, starts, goals, self.explored_nodes, 
                                         timeout=time_budget, w=w,
                                         model=model)
			if optimal_solution is None:
				raise FailureException('cannot find a feasible solution')
			return normal_return()

		except TimeOutException as e:
			print(e)
			optimal_solution = None
			return normal_return()


def focal_cbs(env, starts, goals, explored_nodes, timeout, w, model):

	start_t = time()

	focal_pq = list()
	open_pq = list()
	backplanner = BFSPlanner(env, goals)
	root = Constraint()
	root.h = 0    
	root.find_paths(env, starts, goals, backplanner)
	root.rand = np.random.uniform()
	heapq.heappush(open_pq, ((root.cost, root.rand), root))
	heapq.heappush(focal_pq, ((float('-inf'), 0, 0), root))
	best_f = root.cost
	explored_nodes.append(root)
	while len(open_pq):
		old_best_f = best_f
		best_f = open_pq[0][1].cost
		if old_best_f < best_f:
			for cost_node in open_pq:
				_, node = cost_node
				if (node.cost > w * old_best_f) and (node.cost <= w * best_f):
					heapq.heappush(focal_pq, ((-node.depth, node.h, node.rand), node))
				elif (node.cost > w * best_f):
					break

		if (time()-start_t)>timeout:
			raise TimeOutException("Timeout. Expert fails to find a solution.")

		# root_cost, root_depth, root
		_, x = heapq.heappop(focal_pq)
		open_pq.remove(((x.cost, x.rand), x))
		heapq.heapify(open_pq)

		conflict = find_conflict(env, x.solution)

		if conflict is None:
			if (x.cost != float('inf')):
				x.V = 0
				update_V(env, explored_nodes)
				x.solution = pad_paths(x.solution)
				return x
			else:
				print('CBS failed!')
				return None
			
		else:

			def add_child(child):
				child.find_paths(env, starts, goals, backplanner)
				child.h = model.give_path_prob(child.solution)   
				child.rand = np.random.uniform()                
				heapq.heappush(open_pq, ((child.cost, child.rand), child))
				if child.cost <= w * best_f:
					heapq.heappush(focal_pq, ((-child.depth, child.h, child.rand), child))
				explored_nodes.append(child)

			if conflict['conflict_type'] == 'collided_obstacle':
				child = x.create_child()
				is_new_constraint = child.add_constraint(agent=conflict['agent'], t=conflict['t'], node=conflict['node'])
				# TODO: the behaviour is strange if the following line is uncommented
				# assert is_new_constraint
				add_child(child)
			
			else:
				for agent in ['agent1', 'agent2']:
					child = x.create_child()
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
		if len(node.children) == 0:
			node.V = 1
		else:
			node.V = min([update_V_recursive(env, child) for child in node.children])
	return node.V


def update_V_leaf(env, node):
	node.V = count_conflict(env, node.solution)
	return node.V

class Constraint:

	def __init__(self):
		self.new_agent_constraint = None
		self.constraints = {}
		self.transition_constraints = {}

		self.children = []
		self.solution = None
		self.cost = None
		self.num_constraints_total = 0
		self.depth = 0

		self.V = None

	def create_child(self):
		child = Constraint()
		child.new_agent_constraint = None
		child.constraints = deepcopy(self.constraints)
		child.transition_constraints = deepcopy(self.transition_constraints)
		child.solution = deepcopy(self.solution)
		child.num_constraints_total = self.num_constraints_total
		child.depth = self.depth + 1
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
				path = self.solution[agent]
				paths[agent] = path
			else:
				T = 0
				if (agent in self.constraints) and len(self.constraints[agent]):
					T = max(T, max(self.constraints[agent].keys())+1)
				if (agent in self.transition_constraints) and (len(self.transition_constraints[agent])):
					T = max(T, max(self.transition_constraints[agent].keys())+1)
				path, _, _, _ = astar(env=env, start=starts[agent], goal=goals[agent], agent_id=agent,
																backplanner=backplanner, constraint_fn=self.get_constraint_fn(agent), 
																min_time=T, return_cost=True)
				paths[agent] = path

		
		self.solution = paths
		self.makespan = max([len(p) for p in paths])
		self.flowtime = self.cost = sum([len(p) for p in paths])


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

def count_conflict(env, paths):
	paths = pad_paths(paths)
	num_agents = len(paths)
	num_of_conflict = 0

	for t in range(len(paths[0])):
		# Check collisions (agents on the same node, or inter agent collision)

		for agent_i in range(num_agents):
			v_i_t = paths[agent_i][t]
			v_i_t_pos = env.find_pos_of_node_from_graph(agent_i, v_i_t)

			for agent_j in range(agent_i+1, num_agents):
				v_j_t = paths[agent_j][t]
				v_j_t_pos = env.find_pos_of_node_from_graph(agent_j, v_j_t)

				num_of_conflict += (not env._state_fp(v_i_t_pos)) + (not env._state_fp(v_j_t_pos))
				
				num_of_conflict += env.collide_static_agents(v_i_t_pos, v_j_t_pos)

				if t > 0:
					v_i_t_prev = paths[agent_i][t - 1]
					v_i_t_pos_prev = env.find_pos_of_node_from_graph(agent_i, v_i_t_prev)
					v_j_t_prev = paths[agent_j][t - 1]
					v_j_t_pos_prev = env.find_pos_of_node_from_graph(agent_j, v_j_t_prev)

					num_of_conflict += env.continuous_collide_spheres(v_i_t_pos_prev, v_i_t_pos, v_j_t_pos_prev, v_j_t_pos)
				
				
	return num_of_conflict



