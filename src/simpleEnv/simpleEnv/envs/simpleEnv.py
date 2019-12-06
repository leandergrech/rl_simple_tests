import gym
import numpy as np

from collections import defaultdict
from warnings import warn

from functools import partial


def printWithBorder(string):
	string = f"#   {string}   #"
	hashline = partial(print, *('#' for _ in string), sep='')

	hashline()
	print(string)
	hashline()



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

		self.rm_kwargs = kwargs.get("rm_kwargs", dict(rm_size=5,
													  rm_element_mu=1.5,
													  rm_element_std=0.2,
													  rm_element_clip_low=1.0,
													  rm_element_clip_high=2.0))

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

		self.act_dimension = None
		self.obs_dimension = None
		self._response_matrix = None
		self.reference_trajectory = None

		self.action_space = None
		self.observation_space = None

		'''
		Obtaining a randomised response matrix
		'''
		if kwargs.get('rm_use_last', True):
			try:
				self.response_matrix = np.load(simpleEnv.rm_loc)
				printWithBorder("Loaded response matrix from file")
			except FileNotFoundError:
				printWithBorder("No response matrix found")

		else:
			printWithBorder("Created random response matrix")
			A = self._random_rm()
			np.save(simpleEnv.rm_loc, A)
			self.response_matrix = A

	def _random_rm(self):
		sz, mu, std, lo, hi = self.rm_kwargs.values()
		return np.clip(np.random.normal(mu, std, (sz, sz)), lo, hi)

	@property
	def response_matrix(self):
		return self._response_matrix

	@response_matrix.setter
	def response_matrix(self, rm):
		self._response_matrix = rm
		self.act_dimension = rm.shape[1]
		self.obs_dimension = rm.shape[0]

		min_action, max_action= -10.0, 10.0
		self.action_space = gym.spaces.Box(low=min_action, high=max_action, shape=(self.act_dimension,),
										   dtype=np.float32)

		min_state, max_state = 10.0, 10.0
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

		init_action = self.action_space.sample()
		init_state, init_reward = self._take_action(init_action)
		self.initial_conditions.add_episode()
		self.initial_conditions.push_stat(init_state)

		self.current_state = init_state
		self.last_action = init_action

		return init_state

	def render(self, mode='human', rewards=None):
		if self.fig is None:
			self.fig, ax = plt.subplots(3)
			plt.show(block=False)

			ax[0].plot([0],[0], label="Rewards")
			ax[0].legend(loc='upper left')
			ax[0].set_ylim((-10,1))

			ax[1].plot(self.reference_trajectory, color='k', label= 'Reference Trajectory')
			ax[1].plot(self.current_state, color='b', label="Current state")
			ax[1].set_ylim((self.observation_space.low[0],self.observation_space.high[0]))
			ax[1].legend(loc="upper right")

			ax[2].plot(self.last_action, color ='r', label='Last action')
			ax[2].set_ylim((self.action_space.low[0], self.action_space.high[0]))
			ax[2].legend(loc="upper right")


		axes = self.fig.axes
		ax_rewards = axes[0]
		ax_state = axes[1]
		ax_action = axes[2]

		if rewards:
			max_x = len(rewards)
			if max_x < 50:
				x_list = range(max_x)
			else:
				x_list = np.arange(max_x - 50, max_x)
			rewards = rewards[-50:]
			ax_rewards.get_lines()[0].set_data(x_list, rewards)
			ax_rewards.set_xlim((min(x_list), max(x_list)))

		state_lines = ax_state.get_lines()
		state_lines[0].set_ydata(self.reference_trajectory)
		state_lines[1].set_ydata(self.current_state)
		ax_action.get_lines()[0].set_ydata(self.last_action)

		plt.draw()
		plt.pause(0.05)


class MORsimpleEnv(simpleEnv):
	"""
		Normal matrix decomposition by Singular Value Decomposition (SVD)

		|-------|       |---------------|       |-------|
		|       |       |               |       | *     |
		|       |       |               |       |   *   |       |-------|
		|       |       |               |       |     * |       |       |
		|   A   |   =   |       U       |   x   |   S   |   x   |  V^T  |
		|       |       |               |       |       |       |       |
		|       |       |               |       |       |       |-------|
		|       |       |               |       |       |
		|-------|       |---------------|       |-------|

		   mxn                 mxm                 mxn             nxn

		Reduced matrix by truncating the S matrix

		|-------|       |----|
		|       |       |    |
		|       |       |    |       |-------|       |-------|
		|   ~   |       |  ~ |       |   ~   |       |  V^T  |
		|   A   |   =   |  U |   x   |   S   |   x   |-------|
		|       |       |    |       |-------|
		|       |       |    |
		|       |       |    |
		|-------|       |----|

		   mxn           mxr            rxr             rxn

	"""
	def __init__(self, **kwargs):
		super(MORsimpleEnv, self).__init__(**kwargs)

		n_evals = self.rm_kwargs.get("n_evals", 4)  # Number of eigen values

		U, Sigma, Vt = np.linalg.svd(self.response_matrix)
		Sigma_trunc = Sigma[:n_evals]

		U_trunc = U[:, :n_evals]        # mxr
		S_trunc = np.diag(Sigma_trunc)  # rxr
		Vt_trunc = Vt[:n_evals]         # rxn

		B = np.dot(U_trunc, S_trunc)    # mxr
		A_tilde = np.dot(B, Vt_trunc)   # mxn

		self.response_matrix = B        # mxr
		self.action_decoder = Vt_trunc     # rxn

		self.reduced_response_matrix = np.dot(B, Vt_trunc)

	def actionFromFeature(self, z):
		return np.dot(z, self.action_decoder)


	def testActualModel(self, action):
		actual_action_vector = np.dot(action, self.decoder)
		next_state = np.dot(self.response_matrix_full, actual_action_vector)

		return next_state, self._objective(next_state)


if __name__ == '__main__':
	env=simpleEnv()

	rewards = []
	for _ in range(1000):
		env.reset()
		for _ in range(np.random.choice(1000)):
			_, r, _, _ = env.step(env.action_space.sample())
			rewards.append(r)
			env.render(rewards)

		print(f"resetting")

	# env.initial_conditions.Print()
	# env.states.Print()
	# env.actions.Print()
	# env.rewards.Print()


