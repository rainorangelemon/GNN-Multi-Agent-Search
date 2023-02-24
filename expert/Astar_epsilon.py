from queue import PriorityQueue
from collections import deque
import itertools
import numpy as np
from PIL import Image
import os
import sys
from copy import deepcopy

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

class Node:
	def __init__(self, state, time, fScore, gScore, focalHeuristic):
		self.state = state
		self.time = time
		self.fScore = fScore
		self.gScore = gScore
		self.focalHeuristic = focalHeuristic


def pad_paths(paths):
	paths = deepcopy(paths)
	T = max([len(p) for p in paths])
	paths = [p + [p[-1] for _ in range(T - len(p))] for p in paths]
	return paths


def astar_epsilon(env, start, goal, agent_id, backplanner=None, constraint_fn=lambda node, lastnode, t: True,
				  min_time=0, return_cost=False, paths=None, w=1.1):
	# env is an instance of an Environment subclass
	# start is an instance of node (whatever type you define)
	# end is an instance of node
	# constraint_fn(node, t) returns False if node at time t is invalid

	if isinstance(start, np.ndarray):
		start = tuple(start)
	if isinstance(goal, np.ndarray):
		goal = tuple(goal)

	# reverse dijkstra
	heur = backplanner.distance(agent_id, start, goal)
	oldBestFScore = heur
	BestFScore = heur

	open_set = list()
	close_set = list()
	t = 0

	focal_set = list()
	stateToHeap = dict()

	fScore = heur
	gScore = 0

	Node_start = Node(state=start, time=0, fScore=fScore, gScore=gScore, focalHeuristic=0)
	stateToHeap[(start, 0)] = Node_start

	if backplanner is None:
		backplanner = env

	# similar with came_from
	prevmap = {(start, 0): None}

	open_set.append(Node_start)
	# open_set[start, t] = (fScore, -gScore)  

	# focalHeuristic, lowest fScore,  highest gScore (or lowest -gScore)
	focal_set.append(Node_start)
	assert Node_start in open_set
	assert len(focal_set)<=len(open_set)

	# focal_set[start, t] = (focalHeuristic, fScore, -gScore)

	while len(open_set):
		
		# a focal set from open list sorted by focalHeuristic, lowest fScore and highest gScore (or lowest -gScore)
		best_node = min(open_set, key=lambda node: (node.fScore, -node.gScore))

		oldBestFScore = BestFScore
		BestFScore = best_node.fScore
		if BestFScore > oldBestFScore:
			for node in open_set:
				fScore = node.fScore
				if (fScore > oldBestFScore * w) and (fScore <= BestFScore * w):
					if node not in focal_set:
						focal_set.append(node)
						assert node in open_set
						assert len(focal_set)<=len(open_set)

		cur_node = min(focal_set, key=lambda node: (node.focalHeuristic, node.fScore, -node.gScore))
		(curr, t, focalHeuristic, fScore, gScore) = (cur_node.state, cur_node.time, cur_node.focalHeuristic, cur_node.fScore, cur_node.gScore)

		focal_set.remove(cur_node)
		open_set.remove(cur_node)

		del stateToHeap[curr, t]
		close_set.append((curr, t))

		if (int(curr)==int(goal)) and (t >= min_time):
			if return_cost:
				return construct_path_tuple(prevmap, (curr, t)), cur_node.gScore, BestFScore, cur_node.focalHeuristic
			else:
				return construct_path_tuple(prevmap, (curr, t))

		for child, step_cost in env.get_potential_state_edgecost(agent_id, curr_node=curr):
			
			child_t = t + 1
			tentative_gScore = gScore + 1

			if (child, child_t) in close_set:
				continue

			if not constraint_fn(child, curr, int(child_t)):
				continue

			if (child, child_t) not in stateToHeap:

				fScore_next = tentative_gScore + backplanner.distance(agent_id, child, goal)	
				focalHeuristic_next = focalHeuristic + \
									  focalTransitionHeuristic(env, agent_id, curr, t, child, child_t, paths)  # only transition heuristic is enough
				stateToHeap[(child, child_t)] = Node(state=child,
													 time=child_t,
													 fScore=fScore_next,
													 gScore=tentative_gScore,
													 focalHeuristic=focalHeuristic_next)

				open_set.append(stateToHeap[(child, child_t)])
				if fScore <= BestFScore * w:
					focal_set.append(stateToHeap[(child, child_t)])
					assert stateToHeap[(child, child_t)] in open_set
					assert len(focal_set)<=len(open_set)

			else:			
				cur_node = stateToHeap[(child, child_t)]

				if tentative_gScore >= cur_node.gScore:
					continue

				else:
					last_gScore = cur_node.gScore
					last_fScore = cur_node.fScore
					delta = last_gScore - tentative_gScore
					cur_node.gScore = tentative_gScore
					cur_node.fScore = cur_node.fScore - delta

					if cur_node.fScore <= BestFScore * w and last_fScore > BestFScore * w:
						focal_set.append(cur_node)
						assert cur_node in open_set
						assert len(focal_set)<=len(open_set)

			prevmap[child, child_t] = (curr, t)

	return None


def construct_path_tuple(prevmap, node):
	seq = deque([node])
	# breakpoint()
	while prevmap[node] != None:
		prev = prevmap[node]
		seq.appendleft(prev)
		node = prev
	return [s[0] for s in list(seq)]


def focalTransitionHeuristic(env, agent_i, agent_curState, t_cur, agent_nextState, t_next, paths):
	if paths is None:
		return 0
	paths = pad_paths(paths)
	num_agents = len(paths)
	num_of_conflict = 0

	v_i_t_cur_pos = env.find_pos_of_node_from_graph(agent_i, agent_curState)
	v_i_t_next_pos = env.find_pos_of_node_from_graph(agent_i, agent_nextState)

	for agent_j in range(num_agents):

		if agent_i != agent_j:
			v_j_t = paths[agent_j][min(t_cur, len(paths[agent_j])-1)] 
			v_j_t_cur_pos = env.find_pos_of_node_from_graph(agent_j, v_j_t)

			v_j_t_next = paths[agent_j][min(t_next, len(paths[agent_j])-1)]
			v_j_t_next_pos = env.find_pos_of_node_from_graph(agent_j, v_j_t_next)

			is_conflict = env.continuous_collide_spheres(v_i_t_cur_pos, v_i_t_next_pos, v_j_t_cur_pos, v_j_t_next_pos)

			num_of_conflict += is_conflict

	return num_of_conflict

