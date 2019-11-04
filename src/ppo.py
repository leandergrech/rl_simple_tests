import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import tensorflow as tf

import spinup.algos.ppo.core as core
from spinup.algos.ppo.ppo import PPOBuffer
from spinup.utils.mpi_tools import num_procs
from spinup.utils.mpi_tf import MPIAdamOptimizer, sync_all_params

from simple_env import simpleEnv


class PPO:
	def __init__(self, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
	             train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(), save_freq=10):
		"""

		:param env_fn:  Function which creates copy of the environment. Must satisfy OpenAI Gym API
		:param actor_critic: Function which takes in placeholder symbols for state 'x_ph' and action 'a_ph' and returns main outputs from agent's Tensorflow computation graph: 'pi', 'logp', 'logp_pi', 'v'
			'pi': {Shape: (batch, act_dim)} Samples actions from policy given states
			'logp': {Shape: (batch, )} Gives log probability according to the policy, of taking actions 'a_ph' in states 'x_ph'
			'logp_pi': {Shape: (batch, )} Gives log probability according to the policy, of the action sampled by 'pi'
			'v': {Shape: (batch, )} Gives the value estimate for state in 'x_ph' (Make sure to flatten this)
		:param ac_kwargs(dict): kwargs for the provided actor_critic function provided
		:param seed(int): Seed for random number generator
		:param steps_per_epoch(int): Number of steps of iteraction (s,a) for the agent and the enviroement in each epoch
		:param epochs(int): Number of epochs of iteraction to perform(equivalent to number of policy updates)
		:param gamma(float): Discount factor. Between (0,1)
		:param clip_ratio(float): Hyperparameter for clipping in the policy objective. Roughly how far can the new policy go from the old policy while still improving the objective function. Usually small (0.1, 0.3)
		:param pi_lr(float): Learning rate for policy optimizer
		:param vf_lr: Learning rate for value function optimizer
		:param train_pi_iters(int): Max number of gradient descent steps to take on policy loss per epoch. Note that early stopping may cause the optimiser to take fewer than this.
		:param train_v_iters: Number of gradient descent steps to take on value function per epoch.
		:param lam(float): Lambda for GAE-Lambda. Between (0,1), close to 1.
		:param max_ep_len(int): Max length of trajectory per episode per rollout
		:param target_kl(float): Roughly what KL-divergence is expected between new and old policies after an update. Used for early stopping. Usually small (0.01, 0.05)
		:param logger_kwargs(dict): kwargs for EpochLogger
		:param save_freq(int): How often in terms of gap between epochs to save the current policy and value function.
		"""

		tf.set_random_seed(seed)
		np.random.seed(seed)


		self.env = env_fn()
		self.obs_dim = self.env.observation_space.shape
		self.act_dim = self.env.action_space.shape

		ac_kwargs['action_space'] = self.env.action_space

		# Inputs to computation graph
		self.x_ph, self.a_ph = core.placeholders_from_spaces(self.env.observation_space, self.env.action_space)
		self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

		# Main outputs from computation graph
		self.pi, self.logp, self.logp_pi, self.v = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

		# Need all placeholders in *this* order later (to zip it from buffer)
		self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

		# Every step get: action, value and logprob
		self.get_action_ops = [self.pi, self.v, self.logp_pi]

		# Experience buffer
		self.local_steps_per_epoch = int(steps_per_epoch/num_procs())
		self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)

		# Count variables
		var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])

		# PPO objectives
		self.ratio = tf.exp(self.logp - self.logp_old_ph)    # pi(a|s) / pi_old(a|s)
		self.min_adv = tf.where(self.adv_ph>0, (1+clip_ratio)*self.adv_ph, (1-clip_ratio)*self.adv_ph)
		self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv))
		self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

		# Info
		# TODO

		# Optimizers
		self.train_pi = MPIAdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
		self.train_v = MPIAdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())






	def run(self):
		tf.initialize_all_variables.run()


	def predict(self, state):

	def perceive(self):


