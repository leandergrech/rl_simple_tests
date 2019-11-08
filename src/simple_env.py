import logging

import gym
import numpy as np

# from simple_env_graph import simpleEnvGraph


class simpleEnv(gym.Env):
	"""
	Creates environment using linear response matrix (only diagonals initilised)
	"""

	def __init__(self, **kwargs):
		self.__version__ = "0.0.1"
		logging.info(f"simpleEnv - Version {self.__version__}")
		self.__name__ = f"simpleEnv - Version {self.__version__}"
		self.visualization = None

		self.done = False
		self.MAX_TIME = 25
		self.curr_step = -1

		self.curr_episode = -1
		self.TOTAL_COUNTER = -1
		self.action_episode_memory = []
		self.rewards = []
		self.states = []
		self.actions = []
		self.initial_conditions = []

		self.counter = 0
		self.seed(123)

		self.a_dim_size = 5
		A = np.diag(np.clip(np.random.normal(1.5, 0.2, self.a_dim_size), 1.0, 2.0))

		self.act_dimension = A.shape[0]
		self.obs_dimension = A.shape[1]

		self.MAX_POS = 1

		self.action_space = gym.spaces.Box(low=-self.MAX_POS, high=self.MAX_POS, shape=(self.act_dimension,),
		                                   dtype=np.float32)

		self.observation_space = gym.spaces.Box(low=-self.MAX_POS, high=self.MAX_POS, shape=(self.obs_dimension,),
		                                        dtype=np.float32)

		self.reference_trajectory = np.ones(self.obs_dimension)
		self.response_matrix = A

	def seed(self, seed):
		np.random.seed(seed)

	def step(self, action):
		self.curr_step += 1
		self.counter += 1
		state, reward = self._take_action(action)

		self.action_episode_memory[self.curr_episode].append(action)
		self.rewards[self.curr_episode].append(reward)
		self.states[self.curr_episode].append(state)
		self.actions[self.curr_episode].append(action)
		if reward < -10 or reward > -0.25 or self.curr_step > self.MAX_TIME:
			self.done = True

		return state, reward, self.done, {}

	def _take_action(self, action):
		self.TOTAL_COUNTER += 1
		next_state = np.dot(self.response_matrix, action)
		reward = -np.sqrt(np.mean(np.square(next_state - self.reference_trajectory)))

		return next_state, reward

	def reset(self):
		self.curr_episode += 1
		self.curr_step = 0

		self.action_episode_memory.append([])
		self.rewards.append([])
		self.states.append([])
		self.actions.append([])

		self.done = False
		init_state, init_reward = self._take_action(np.random.randn(self.act_dimension))
		self.initial_conditions.append(init_state)

		return init_state

	def render(self, mode='human'):
		from simple_env_graph import simpleEnvGraph
		if self.visualization is None:
			self.visualization = simpleEnvGraph(title="simpleEnv", adaptive_lims=True)

		self.visualization.render(self.curr_step, self.rewards[-1][-1], self.states[-1][-1], self.actions[-1][-1])


if __name__ == '__main__':
	env = simpleEnv()

	import matplotlib.pyplot as plt

	for i in range(1000):
		env.reset()
		action = np.random.rand(env.act_dimension)
		state, _, _, _ = env.step(action)
		# state_line.set_ydata(state)
		env.render()

		plt.pause(0.01)

	input()
