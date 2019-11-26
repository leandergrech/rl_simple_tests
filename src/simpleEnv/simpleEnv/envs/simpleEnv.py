import gym
import numpy as np

from collections import defaultdict
from warnings import warn


class epochStats:
	def __init__(self, name):
		self.name = name
		self.stats = []
		self.__current_ep = -1

	def add_episode(self):
		try:
			if len(self.stats[-1]) == 0:
				warn(f"Previous '{self.name}' epoch stat is empty")
		except IndexError:
			pass

		self.stats.append([])
		self.__current_ep += 1

	def push_stat(self, stat):
		if len(self.stats) == 0:
			self.add_episode()

		self.stats[self.__current_ep].append(stat)

	def push_stat_and_add_episode(self, stat):
		self.push_stat(stat)
		self.add_episode()


	def Print(self):
		if self.__current_ep == -1:
			print(f"No stats in '{self.name}' epochStats instance")
			return

		print(f"{self.name}")
		for i, ep_stat in enumerate(self.stats):
			print(f"Episode {i} -> ", end='\t')
			for item in ep_stat:
				print(f"{item},", end='\t')
			print()

import matplotlib.pyplot as plt
plt.ion()

class simpleEnv(gym.Env):
	rm_loc = 'simpleEnv_response_matrix.npy'
	def __init__(self, **kwargs):
		'''
		Initialising parameters
		'''
		self.current_state = None
		self.last_action = None

		self.done = False
		self.MAX_TIME = 40
		self.curr_step = -1

		self.curr_episode = -1
		self.rewards = epochStats(name="rewards")
		self.states = epochStats(name="states")
		self.actions = epochStats(name="actions")
		self.initial_conditions = epochStats(name="initial_conditions")


		self.reward_threshold = kwargs.get('reward_threshold', -0.2)

		self.fig = None

		'''
		Obtaining a randomised response matrix
		'''
		def create_response_matrix():
			nonlocal kwargs
			rm_size = kwargs.get('rm_size', 5)
			rm_element_mu = kwargs.get('rm_element_mu', 1.5)
			rm_element_std = kwargs.get('rm_element_std', 0.2)
			rm_element_clip_low = kwargs.get('rm_element_clip_low', 1.0)
			rm_element_clip_high = kwargs.get('rm_element_clip_high', 2.0)
			A = np.clip(np.random.normal(rm_element_mu, rm_element_std, (rm_size, rm_size)), rm_element_clip_low,
			                    rm_element_clip_high)
			np.save(simpleEnv.rm_loc, A)

			return A

		rm_use_last = kwargs.get('rm_use_last', True)
		if rm_use_last:
			try:
				A = np.load(simpleEnv.rm_loc)
			except FileNotFoundError:
				print("############################################################")
				print("#    No response matrix found. Creating new one instead    #")
				print("############################################################")
				A = create_response_matrix()
		else:
			A = create_response_matrix()



		'''
		Action and observations Spaces
		'''
		self.act_dimension = A.shape[0]
		self.obs_dimension = A.shape[1]
		self.response_matrix = A

		min_action = -10.0
		max_action = 10.0
		self.action_space = gym.spaces.Box(low=min_action, high=max_action, shape=(self.act_dimension,),
		                                   dtype=np.float32)

		min_state = -1.0
		max_state = 1.0
		self.observation_space = gym.spaces.Box(low=min_state, high=max_state, shape=(self.obs_dimension,),
		                                        dtype=np.float32)

		self.reference_trajectory = np.ones(self.obs_dimension)

	def step(self, action):
		self.curr_step += 1
		next_state, reward = self._take_action(action)

		self.last_action = action
		self.current_state = next_state

		for epochStat, stat in zip([self.rewards, self.states, self.actions], [reward, next_state, action]):
			epochStat.push_stat(stat)

		# TODO check if lower reward threshold should be set as well
		if reward > self.reward_threshold or self.curr_step > self.MAX_TIME:
			self.done = True

		return next_state, reward, self.done, {}

	def _objective(self, next_state):
		return -np.sqrt(np.mean(np.square(next_state - self.reference_trajectory)))

	def _take_action(self, action):
		next_state = np.dot(self.response_matrix, action)
		reward = self._objective(next_state=next_state)

		return next_state, reward

	def reset(self):
		self.curr_episode += 1
		self.curr_step = 0

		for item in [self.rewards, self.states, self.actions]:
			item.add_episode()

		self.done = False

		init_state, init_reward = self._take_action(self.action_space.sample())
		self.initial_conditions.add_episode()
		self.initial_conditions.push_stat(init_state)

		return init_state

	def render(self, mode='human', rewards=None):
		if self.fig is None:
			self.fig, ax = plt.subplots(3)
			plt.show(block=False)
		axes = self.fig.axes
		ax_rewards = axes[0]
		ax_state = axes[1]
		ax_action = axes[2]

		for ax in axes:
			ax.clear()

		if rewards:
			max_x = len(rewards)
			if max_x < 50:
				x_list = range(max_x)
			else:
				x_list = np.arange(max_x - 50, max_x)
			rewards = rewards[-50:]
			ax_rewards.plot(x_list, rewards, label="Rewards")


		ax_state.plot(self.reference_trajectory, color='k', label= 'Reference Trajectory')
		ax_state.plot(self.current_state, color='b', label="Current state")
		ax_state.set_ylim((-10,10))
		ax_state.legend(loc="upper right")

		ax_action.plot(self.last_action, color ='r', label='Last action')
		ax_action.set_ylim((self.action_space.low[0], self.action_space.high[0]))
		ax_action.legend(loc="upper right")

		plt.draw()
		plt.pause(0.05)


class MORsimpleEnv(simpleEnv):
	def __init__(self, **kwargs):
		super(MORsimpleEnv, self).__init__(**kwargs)

		rm_output_size = kwargs.get('rm_output_size', 4)
		rm_input_size = kwargs.get('rm_input_size', 5)
		rm_element_mu = kwargs.get('rm_element_mu', 1.5)
		rm_element_std = kwargs.get('rm_element_std', 0.2)
		rm_element_clip_low = kwargs.get('rm_element_clip_low', -1.0)
		rm_element_clip_high = kwargs.get('rm_element_clip_high', 1.0)

		# A = np.pad(np.diag(np.clip(np.random.normal(rm_element_mu, rm_element_std, rm_output_size), rm_element_clip_low,
		# 						   rm_element_clip_high)), ((0, rm_input_size - rm_output_size), (0, 0)))

		A = np.random.uniform(rm_element_clip_low, rm_element_clip_high, (rm_input_size, rm_output_size))

		n_evals = kwargs.get("n_evals", 4)  # Number of eigen values

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
	env=simpleEnv()

	for _ in range(np.random.choice(100)):
		env.reset()
		for _ in range(np.random.choice(10)):
			env.step(np.random.uniform(-1,1,5))
			env.render()

	# env.initial_conditions.Print()
	# env.states.Print()
	# env.actions.Print()
	# env.rewards.Print()

