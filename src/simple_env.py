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

		rm_output_size = kwargs.get('rm_output_size', 6)
		rm_input_size = kwargs.get('rm_input_size', 20)
		rm_element_mu = kwargs.get('rm_element_mu', 1.5)
		rm_element_std = kwargs.get('rm_element_std', 0.2)
		rm_element_clip_low = kwargs.get('rm_element_clip_low', 1.0)
		rm_element_clip_high = kwargs.get('rm_element_clip_high', 2.0)

		A = np.pad(np.diag(np.clip(np.random.normal(rm_element_mu, rm_element_std, rm_output_size), rm_element_clip_low,
								   rm_element_clip_high)), ((0, rm_input_size - rm_output_size), (0, 0)))

		n_evals = kwargs.get("n_evals", 6)  # Number of eigen values

		U, Sigma, Vt = np.linalg.svd(A)
		Sigma_trunc = Sigma[:n_evals]

		S_trunc = np.diag(Sigma_trunc)
		U_trunc = U[:, :n_evals]
		Vt_trunc = Vt[:n_evals]

		B = np.dot(U_trunc, S_trunc)
		A_tilde = np.dot(B, Vt_trunc)

		self.act_dimension = B.shape[1]
		self.obs_dimension = B.shape[0]
		self.response_matrix = B
		self.response_matrix_reduced = A_tilde
		self.response_matrix_full = A
		self.decoder = Vt_trunc

		self.MAX_POS = 1

		action_pos_factor = 2
		self.action_space = gym.spaces.Box(low=-action_pos_factor * self.MAX_POS, high=action_pos_factor * self.MAX_POS,
										   shape=(self.act_dimension,), dtype=np.float32)

		state_pos_factor = 1
		self.observation_space = gym.spaces.Box(low=-state_pos_factor * self.MAX_POS,
												high=state_pos_factor * self.MAX_POS, shape=(self.obs_dimension,),
												dtype=np.float32)

		self.reference_trajectory = np.ones(self.obs_dimension)

	def _objective(self, temp):
		return -np.sqrt(np.mean(np.power(np.subtract(self.reference_trajectory, temp), 2)))

	def testActualModel(self, action):
		actual_action_vector = np.dot(action, self.decoder)
		next_state = np.dot(self.response_matrix_full, actual_action_vector)

		return next_state, self._objective(next_state)




if __name__ == '__main__':
	from time import time
	np.random.seed(int(time()))
	env = MORsimpleEnv()
	env.reset()

	max_state_val = None
	min_state_val = None

	from tqdm import tqdm as progress
	for _ in progress(range(10000)):


		input_vec = np.random.uniform(-1,1, env.act_dimension)
		s, _, _, _ = env.step(input_vec)

		min_this = np.min(s)
		max_this = np.max(s)

		if max_state_val is None:
			max_state_val = max_this
		else:
			max_state_val = max(max_state_val, max_this)

		if min_state_val is None:
			min_state_val = min_this
		else:
			min_state_val = min(max_state_val, max_this)

	from time import sleep
	sleep(0.1)
	print(min_state_val, max_state_val)






	'''
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots()
	plt.ion()
	plt.show(block=False)

	rmse = []
	rmse_std = []
	n_evals = []
	for i in np.arange(550, 0, -10):
		env = MORsimpleEnv(rm_input_size=1150, rm_output_size=550, n_evals=i)

		rmse_temp = []
		for step in range(100):
			input_vec = np.random.uniform(-1, 1, 550)

			from_reduced = env.response_matrix_reduced.dot(input_vec)
			from_full = env.response_matrix_full.dot(input_vec)

			rmse_temp.append(np.sqrt(np.mean(np.power(np.subtract(from_full, from_reduced), 2))))

		std = np.std(rmse_temp)
		mu = np.mean(rmse_temp)
		rmse.append(mu)
		rmse_std.append(std)
		n_evals.append(i)

		ax.clear()
		ax.plot(n_evals, rmse, label='RMSE')
		plt.yscale("log")
		ax.fill_between(n_evals, np.subtract(rmse, rmse_std), np.add(rmse, rmse_std), facecolor='#9999ff', alpha=0.5)

		ax.legend(loc="best")

		plt.pause(0.1)

	plt.show(block=True)
	'''
