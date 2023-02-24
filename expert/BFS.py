from collections import defaultdict
# this class serves as the backward heuristic for space-time A*

class BFSPlanner:
    def __init__(self, env, goals):
        self.travel_times = [{int(goal): 0} for goal in goals]
        self.queues = [[goal] for goal in goals]
        self.env = env
    
    def distance(self, agent_id, child, goal):
        if child in self.travel_times[agent_id]:
            return self.travel_times[agent_id][child]
        else:
            self.bfs(agent_id, child)
            return self.travel_times[agent_id][child]

    def bfs(self, agent_id, target):
        while (target not in self.travel_times[agent_id]):
            curr = self.queues[agent_id].pop(0)
            for child, step_cost in self.env.get_potential_state_edgecost(agent_id, curr_node=curr):
                if child not in self.travel_times[agent_id]:
                    self.travel_times[agent_id][child] = self.travel_times[agent_id][curr] + 1
                    self.queues[agent_id].append(child)
