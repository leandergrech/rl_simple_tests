import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict


class simpleEnvGraph:
	"""
	Visualisation tool for LHCEnv Gym envirnoment
	"""
	fig = plt.figure()
	reward_ax = plt.subplot2grid((11, 1), (0, 0), rowspan=2, colspan=1)
	state_ax = plt.subplot2grid((11, 1), (3, 0), rowspan=3, colspan=1)
	action_ax = plt.subplot2grid((11, 1), (7, 0), rowspan=3, colspan=1)

	def __init__(self, **kwargs):
		# Instance metaData
		self.adaptive_lims = kwargs.get('adaptive_lims', False)
		self.window_size = kwargs.get('window_size', 1000)

		# self.fig = plt.figure()
		# self.fig.suptitle(kwargs.get('title', ""))

		# # Initialise axes
		# self.reward_ax = kwargs.get('reward_ax', plt.subplot2grid((11, 1), (0, 0), rowspan=2, colspan=1))
		# self.state_ax = kwargs.get('state_ax', plt.subplot2grid((11, 1), (3, 0), rowspan=3, colspan=1))
		# self.action_ax = kwargs.get('action_ax', plt.subplot2grid((11, 1), (7, 0), rowspan=3, colspan=1))

		# Adjust plot and matplotlib settings
		# if kwargs.get('enable_plt_setup', True):
		plt.subplots_adjust(left=0.075, bottom=0.02, right=0.975, top=0.93, wspace=0.185, hspace=0.1)
		plt.ion()
		plt.show()

		# Initialise reward, state and action members
		self.rewards = np.array([])
		self.state = None
		self.action = None

	@staticmethod
	def adaptive_limits(y_data):
		"""
		Add top and bottom space to the y_lim and returns y_lims
		:param y_data: Should be array-like
		"""
		min_ylim = min(y_data)
		max_ylim = max(y_data)
		ylim_range = abs(max_ylim - min_ylim)
		ylim_range = ylim_range if ylim_range > 0 else 1  # Avoid y_lim same size warning

		min_ylim = min_ylim - ylim_range / 10
		max_ylim = max_ylim + ylim_range / 10

		return min_ylim, max_ylim

	def get_moving_window(self, current_step, data):
		min_xrange = current_step - self.window_size
		xrange = np.linspace(min_xrange if min_xrange > 0 else 0, current_step - 1, len(data))

		return xrange

	def clip_reward_length(self, reward_list):
		return np.delete(reward_list, np.s_[:-self.window_size])

	def render(self, current_step, reward, state, action):
		self.rewards = self.clip_reward_length(np.append(self.rewards, reward))
		self.state = state
		self.action = action

		self.clearAll()
		self._render_reward(current_step)
		self._render_state()
		self._render_action()
		self.render_decorations()
		self.render_legends()

		plt.pause(0.1)

	def _render_reward(self, current_step):
		self.reward_ax.plot(self.get_moving_window(current_step=current_step, data=self.rewards), self.rewards,
							color='k', label='Rewards')

		if self.adaptive_lims:
			self.reward_ax.set_ylim(self.adaptive_limits(self.rewards))

	def _render_state(self):
		self.state_ax.plot(range(len(self.state)), self.state, color='b', label='State')

		if self.adaptive_lims:
			self.state_ax.set_ylim(self.adaptive_limits(self.state))

	def _render_action(self):
		self.action_ax.plot(range(len(self.action)), self.action, color='r', label='Action')

		if self.adaptive_lims:
			self.action_ax.set_ylim(self.adaptive_limits(self.action))
		else:
			self.action_ax.set_ylim((-0.05, 0.05))

	def render_decorations(self):
		self.reward_ax.set_xlabel('Steps')
		self.state_ax.set_ylabel('BPM error offset [um]')
		self.action_ax.set_ylabel('COD deflection trim [urad]')

	def render_legends(self):
		self.state_ax.legend(loc="upper left")
		self.action_ax.legend(loc="upper left")
		self.reward_ax.legend(loc="upper left")

	def clearAll(self):
		self.action_ax.clear()
		self.state_ax.clear()
		self.reward_ax.clear()

if __name__ == '__main__':

	dummy_state = np.random.uniform(-1, 1, 10)
	dummy_action = np.random.uniform(0, 0.1, 15)
	tv = simpleEnvGraph(adaptive_lims=True)
	tv2 = simpleEnvGraph()

	plt.subplots_adjust(left=0.075, bottom=0.02, right=0.975, top=0.93, wspace=0.185, hspace=0.1)
	plt.ion()
	plt.show()

	for i in range(10):

		tv.render(i, i, dummy_state, dummy_action)
		tv2.render(i, i + np.random.uniform(-1, 1), -dummy_state, -dummy_action)
		tv.render_legends()


		print('iteration', i)

		plt.pause(1)
		tv.clearAll()

