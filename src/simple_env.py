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
		self.MAX_TIME = 50
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

		rm_size = kwargs.get('rm_size', 5)
		rm_element_mu = kwargs.get('rm_element_mu', 1.5)
		rm_element_std = kwargs.get('rm_element_std', 0.2)
		rm_element_clip_low = kwargs.get('rm_element_clip_low', 1.0)
		rm_element_clip_high = kwargs.get('rm_element_clip_high', 1.0)
		A = np.diag(np.clip(np.random.normal(rm_element_mu, rm_element_std, rm_size), rm_element_clip_low,
							rm_element_clip_high))

		self.act_dimension = A.shape[0]
		self.obs_dimension = A.shape[1]
		self.response_matrix = A

		self.MAX_POS = 1

		action_pos_factor = 2
		self.action_space = gym.spaces.Box(low=-action_pos_factor * self.MAX_POS, high=action_pos_factor * self.MAX_POS,
										   shape=(self.act_dimension,), dtype=np.float32)

		state_pos_factor = 1
		self.observation_space = gym.spaces.Box(low=-state_pos_factor * self.MAX_POS,
												high=state_pos_factor * self.MAX_POS, shape=(self.obs_dimension,),
												dtype=np.float32)

		self.reference_trajectory = np.ones(self.obs_dimension)

	def seed(self, seed):
		np.random.seed(seed)

	def _stepThroughModel(self, action):
		self.curr_step += 1
		self.counter += 1
		state, reward = self._take_action(action)

		self.action_episode_memory[self.curr_episode].append(action)
		self.rewards[self.curr_episode].append(reward)
		self.states[self.curr_episode].append(state)
		self.actions[self.curr_episode].append(action)
		if reward < -15 or reward > -0.2 or self.curr_step > self.MAX_TIME:
			self.done = True

		return state, reward, self.done, {}

	def step(self, action):
		return self._stepThroughModel(action)

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

class MORsimpleEnv(simpleEnv):
	def __init__(self, **kwargs):
		super(MORsimpleEnv, self).__init__(**kwargs)
		
		rm_size = kwargs.get('rm_size', 5)
		rm_element_mu = kwargs.get('rm_element_mu', 1.5)
		rm_element_std = kwargs.get('rm_element_std', 0.2)
		rm_element_clip_low = kwargs.get('rm_element_clip_low', 1.0)
		rm_element_clip_high = kwargs.get('rm_element_clip_high', 1.0)

		A = np.diag(np.clip(np.random.normal(rm_element_mu, rm_element_std, rm_size), rm_element_clip_low,
							rm_element_clip_high))

		U,S,Vt = np.linalg.svd(A)

		n_evals = kwargs.get("n_evals", rm_size)


if __name__ == '__main__':
	env = simpleEigenEnv(n_eigenvalues=4)

	import matplotlib.pyplot as plt

	for i in range(1000):
		env.reset()
		action = np.random.rand(env.act_dimension)
		state, _, _, _ = env.step(action)
		# state_line.set_ydata(state)
		env.render()

		plt.pause(0.01)

	input()
