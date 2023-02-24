import pygame
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Colour(Enum):
	Black = (0,0,0)
	Grey = (100,100,100)
	White = (255,255,255)
	Red = (168,22,0)
	Blue = (22,0,168)
	Green = (0,168,22)
	Cyan = (0,168,168)
	Yellow = (200,200,0)
	Magenta = (168,0,168)
	Orange = (200,90,0)
	Purple = (100,0,168)
	LightRed = (200,100,100)
	LightBlue = (100,100,200)
	LightGreen = (100,200,100)
	LightCyan = (100,200,200)
	LightYellow = (200,200,100)
	LightMagenta = (200,100,200)
	LightPurple = (168,100,255)
	LightOrange = (255,168,100)


colour_order = [Colour.Red, Colour.Blue, Colour.Green, Colour.Cyan, Colour.Yellow, Colour.Magenta, Colour.Orange, Colour.Purple,
				Colour.LightRed, Colour.LightBlue, Colour.LightGreen, Colour.LightCyan, Colour.LightYellow, Colour.LightMagenta, Colour.LightOrange, Colour.LightPurple]


class Visualiser:

	def __init__(self):
		self.HEIGHT = 800
		self.GAP_SIZE = 1
		self.initialised = False

	def setup(self, shape):
		self.shape = shape
		pygame.init()
		self.WIDTH = self.HEIGHT * int(self.shape[0] / self.shape[1])
		self.BLOCK_HEIGHT = int((self.HEIGHT - ((self.shape[1]+1) * self.GAP_SIZE)) / self.shape[1])
		self.BLOCK_WIDTH = int((self.WIDTH - ((self.shape[0]+1) * self.GAP_SIZE)) / self.shape[0])
		self.screen = pygame.display.set_mode((self.HEIGHT, self.WIDTH))
		self.clock = pygame.time.Clock()
		self.screen.fill(Colour.Black.value)

	def init_grid(self):
		self.grid_squares = [[pygame.Rect(x*(self.BLOCK_WIDTH+self.GAP_SIZE)+self.GAP_SIZE, self.HEIGHT-self.BLOCK_HEIGHT-y*(self.BLOCK_HEIGHT+self.GAP_SIZE)+self.GAP_SIZE, self.BLOCK_WIDTH, self.BLOCK_HEIGHT) for y in range(self.shape[1])] for x in range(self.shape[0])]
		self.grid_pos = [[(int((x+1)*self.GAP_SIZE+(x+0.5)*self.BLOCK_WIDTH), self.HEIGHT-int((y+1)*self.GAP_SIZE+(y+0.5)*self.BLOCK_HEIGHT)) for y in range(self.shape[1])] for x in range(self.shape[0])]

	def clear(self):
		for x in range(self.shape[0]):
			for y in range(self.shape[1]):
				rect = self.grid_squares[x][y]
				pygame.draw.rect(self.screen, Colour.White.value, rect, 0)

	def draw_agents(self, positions):
		for i, pos in enumerate(positions):
			coord = self.grid_pos[pos[0]][pos[1]]
			colour = colour_order[i].value
			radius = int(0.9 * min(self.BLOCK_WIDTH, self.BLOCK_HEIGHT) / 2)
			pygame.draw.circle(self.screen, colour, coord, radius, 0)
			pygame.draw.circle(self.screen, Colour.Black.value, coord, radius, 2)

	def draw_goals(self, goals):
		if isinstance(goals, np.ndarray):
			for i, goal in enumerate(goals):
				colour = colour_order[i].value
				rect = self.grid_squares[goal[0]][goal[1]]
				pygame.draw.rect(self.screen, colour, rect, 0)
		elif isinstance(goals, dict):
			for goal, i in goals.items():
				if i == -1:
					colour = Colour.Grey.value
				else:
					colour = colour_order[i].value
				rect = self.grid_squares[goal[0]][goal[1]]
				pygame.draw.rect(self.screen, colour, rect, 0)

	def draw_obstacles(self, grid):
		x_coords, y_coords = np.where(grid)
		for x, y in zip(x_coords, y_coords):
			rect = self.grid_squares[x][y]
			pygame.draw.rect(self.screen, Colour.Black.value, rect, 0)


	def render(self, grid, positions, goals=None):
		if not self.initialised:
			self.setup(grid.shape)
			self.init_grid()
			self.initialised = True
		self.clear()
		self.draw_obstacles(grid)
		self.draw_goals(goals)
		self.draw_agents(positions)
		pygame.display.flip()
		for event in pygame.event.get(): 
			if event.type == pygame.QUIT: 
				pygame.quit()


def plot_v_and_h(env, id_map, points, edge_indexeses, titles):
	env.load_map(id_map)
	plt.clf()
	plt.close('all')
	num_graph = len(titles)
	fig = plt.figure(figsize=(10*num_graph, 10))
	for id_graph, edge_indexes, title in zip(range(num_graph), edge_indexeses, titles):
		ax = fig.add_subplot(1, num_graph, id_graph+1)

		if env.config.env_name == 'MazeEnv':
			env.plot_polygon(env.occupied_area, ax=ax, alpha=1.0, fc='#253494', ec='#253494')
		elif env.config.env_name == 'BoxEnv':
			env.plot_polygon(env.occupied_area, ax=ax, alpha=1.0, fc='saddlebrown', ec='none')

		src_pos = points[edge_indexes[:, 0]]
		end_pos = points[edge_indexes[:, 1]]
		pos_xy_pos = list(zip(src_pos, end_pos))

		lines = LineCollection(np.array(pos_xy_pos), linestyle='solid')
		ax.add_collection(lines)

		ax.set_title(title)

	fig = plt.gcf()
	fig.canvas.draw()

	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	return img





