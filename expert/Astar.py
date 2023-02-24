from queue import PriorityQueue
from collections import deque
import itertools
import numpy as np
from PIL import Image

STEP_SIZE = 4.1e-2
# MAX_STEPS = MAX_EDGE_COST


def astar(env, start, goal, agent_id, backplanner=None, constraint_fn=lambda node, lastnode, t: True, min_time=0, return_cost=False):
	# env is an instance of an Environment subclass
	# start is an instance of node (whatever type you define)
	# end is an instance of node
	# constraint_fn(node, t) returns False if node at time t is invalid

	if isinstance(start, np.ndarray):
		start = tuple(start)
	if isinstance(goal, np.ndarray):
		goal = tuple(goal)

	pq = dict()
	cost = 0.0
	t = 0.0

	if backplanner is None:
		backplanner = env

	# reverse dijkstra
	heur = backplanner.distance(agent_id, start, goal)

	costmap = {(start, 0): cost}
	prevmap = {(start, 0): None}
	best = (None, float('inf'))
	pq[start, t] = heur+cost
	while len(pq):
		curr, t = min(pq, key=pq.get)
		totcost = pq[curr, t]

		del pq[curr, t]
		if totcost < best[1]:
			best = ((curr, t), totcost)

		if (int(curr)==int(goal)) and (t >= min_time):  # TODO: need to be fixed later to 'and t > min_t'
			# print('i"m in goal')
			if return_cost:
				return construct_path_tuple(prevmap, (curr, t)), totcost, set(), set()
			else:
				return construct_path_tuple(prevmap, (curr, t)), set(), set()

		for child, step_cost in env.get_potential_state_edgecost(agent_id, curr_node=curr):
			
			child_t = t + 1

			# print(agent_id, curr, child, child_t)

			if not constraint_fn(child, curr, int(child_t)):
				# print(f'find constraint - {child} - {curr} - {child_t} -{env.get_potential_state_edgecost(agent_id, curr_node=curr)}')
				continue

			child_cost = costmap.get((curr, t), float('inf')) + 1
			if child_cost < costmap.get((child, child_t), float('inf')):
				prevmap[child, child_t] = (curr, t)
				costmap[child, child_t] = child_cost
				child_totcost = backplanner.distance(agent_id, child, goal) + child_cost  # TODO: change distance to BFS-based heuristic
				pq[child, child_t] = child_totcost
	if return_cost:
		return construct_path_tuple(prevmap, best[0]), float('inf'), set(), set()
	else:
		return construct_path_tuple(prevmap, best[0]), set(), set()





def combine_actions(node_list):
	return itertools.product(*node_list)

def construct_path_time(prevmap, node, t):
	seq = deque([(node, t)])
	# breakpoint()
	while prevmap[node] != None:
		prev = prevmap[node]
		seq.appendleft(prev)

		node = prev[0]

	return list(seq)

def construct_path(prevmap, node):
	seq = deque([node])
	# breakpoint()
	while prevmap[node] != None:
		prev = prevmap[node]
		seq.appendleft(prev)
		node = prev
	return list(seq)


def construct_path_tuple(prevmap, node):
	seq = deque([node])
	# breakpoint()
	while prevmap[node] != None:
		prev = prevmap[node]
		seq.appendleft(prev)
		node = prev
	return [s[0] for s in list(seq)]


def construct_path_multi(prevmap, nodes):
	seqs = [deque([node]) for node in nodes]
	while prevmap[nodes] != None:
		prevs = prevmap[nodes]
		for i, prev in enumerate(prevs):
			seqs[i].appendleft(prev)
		nodes = prevs
	return [list(seq) for seq in seqs]



