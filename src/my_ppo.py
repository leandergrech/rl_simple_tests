import os

dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import tensorflow as tf
import time
from datetime import datetime as dt
import argparse

import spinup.algos.ppo.core as core
from spinup.algos.ppo.ppo import PPOBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import num_procs, mpi_avg, mpi_fork
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params

from simple_env import MORsimpleEnv

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

class PPO:
	def __init__(self, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000,
				 epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
				 train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(),
				 save_freq=10):
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

		self.steps_per_epoch = steps_per_epoch
		self.epochs = epochs
		self.train_pi_iters = train_pi_iters
		self.train_v_iters = train_v_iters
		self.max_ep_len = max_ep_len

		self.target_kl = target_kl
		self.save_freq = save_freq
		self.logger = EpochLogger(**logger_kwargs)
		self.logger.save_config(locals())

		ac_kwargs['action_space'] = self.env.action_space

		# Inputs to computation graph
		self.x_ph, self.a_ph = core.placeholders_from_spaces(self.env.observation_space, self.env.action_space)
		self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

		# Main outputs from computation graph
		self.pi, self.logp, self.logp_pi, self.v = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)
		# tf.keras.Sequential

		# Need all placeholders in *this* order later (to zip it from buffer)
		self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

		# Every step get: action, value and logprob
		self.get_action_ops = [self.pi, self.v, self.logp_pi]

		# Experience buffer
		# self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
		self.local_steps_per_epoch = steps_per_epoch
		self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)

		# Count variables
		var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])

		# PPO objectives
		self.ratio = tf.exp(self.logp - self.logp_old_ph)  # pi(a|s) / pi_old(a|s)
		self.min_adv = tf.where(self.adv_ph > 0, (1 + clip_ratio) * self.adv_ph, (1 - clip_ratio) * self.adv_ph)
		self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv))
		self.v_loss = tf.reduce_mean((self.ret_ph - self.v) ** 2)

		# Info
		## a sample estimate for KL-divergence, easy to compute
		self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)
		## a sample estimate for entropy, also easy to compute
		self.approx_ent = tf.reduce_mean(-self.logp)
		self.clipped = tf.logical_or(self.ratio > (1 + clip_ratio), self.ratio < (1 - clip_ratio))
		self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32))

		# Optimizers
		# self.train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
		# self.train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)
		self.train_pi = tf.compat.v1.train.AdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
		self.train_v = tf.compat.v1.train.AdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)

		config = tf.compat.v1.ConfigProto()
		config.intra_op_parallelism_threads = 8
		config.inter_op_parallelism_threads = 8
		self.sess = tf.Session(config=config)

		# Initialize all the variables defined here
		self.sess.run(tf.global_variables_initializer())

		# Sync parameters across processes
		# self.sess.run(sync_all_params())

		# Tensorboard setting up
		self.summ_writer = tf.summary.FileWriter(self._setup_summary_dir(), self.sess.graph)

		# Summary placeholders
		with tf.name_scope('loss'):
			self.pi_loss_ph = tf.placeholder(dtype=tf.float32, shape=None, name='pi_loss_summary')
			self.v_loss_ph = tf.placeholder(dtype=tf.float32, shape=None, name='v_loss_summary')

			self.pi_loss_summary = tf.summary.scalar('pi_loss', self.pi_loss_ph)
			self.v_loss_summary = tf.summary.scalar('v_loss', self.v_loss_ph)

		with tf.name_scope('performance'):
			self.accuracy_ph = tf.placeholder(dtype=tf.float32, shape=None, name='accuracy_summary')

			self.accuracy_summary = tf.summary.scalar('rms_error', self.accuracy_ph)

		self.loss_summaries = tf.summary.merge([self.pi_loss_summary, self.v_loss_summary])

	def _setup_summary_dir(self):
		# TODO Check how many summaries exist with the same name and append a counter to the name
		summary_dir = f"ppo-{self.epochs}epochs-{self.steps_per_epoch}steps-{dt.now().strftime('%d%m%y%H%M%S')}"
		summary_path = "summaries"

		if not os.path.exists(summary_path):
			os.makedirs(summary_path)

		return os.path.join(summary_path, summary_dir)

	def update(self):
		inputs = {k: v for k, v in zip(self.all_phs, self.buf.get())}
		pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

		# Training
		for i in range(self.train_pi_iters):
			_, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
			# kl = mpi_avg(kl)
			if kl > 1.5 * self.target_kl:
				self.logger.log(f"Early stopping at step {i} due to reaching max kl.", color='crimson')
				break
		self.logger.store(StopIter=i)
		for _ in range(self.train_v_iters):
			self.sess.run(self.train_v, feed_dict=inputs)

		# Log changes from update
		pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.approx_ent],
												  feed_dict=inputs)
		self.logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl, Entropy=ent, ClipFrac=cf,
						  DeltaLossPi=(pi_l_new - pi_l_old), DeltaLossV=(v_l_new - v_l_old))

		return pi_l_new, v_l_new

	def run(self):
		start_time = time.time()
		o = self.env.reset()
		r, d, ep_ret, ep_len = 0, False, 0, 0

		# Collect experience form env and update/log each epoch
		for epoch in range(self.epochs):
			for t in range(self.local_steps_per_epoch):
				# print(f"run step {t}")
				a, v_t, logp_t = self.sess.run(self.get_action_ops, feed_dict={self.x_ph: o.reshape(1, -1)})

				# Save and log
				self.buf.store(o, a, r, v_t, logp_t)
				self.logger.store(VVals=v_t)

				o, r, d, _ = self.env.step(a[0])
				ep_ret += r
				ep_len += 1

				terminal = (d or (ep_len == self.max_ep_len))
				if terminal or (t == self.local_steps_per_epoch - 1):
					if not terminal:
						self.logger.log(f"Warning, trajectory cut off by epoch at {ep_len} steps.", 'yellow')
					# If trajectory did not reach terminal state, bootstrap value target
					last_val = r if d else self.sess.run(self.v, feed_dict={self.x_ph: o.reshape(1, -1)})
					self.buf.finish_path(last_val)
					if terminal:
						# Only save EpRet / EpLen if trajectory is finished
						self.logger.store(EpRet=ep_ret, EpLen=ep_len)
					o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

			# Save model
			# if epoch % self.save_freq == 0 or epoch == self.epochs - 1:
			# if epoch == self.epochs - 1:
			# 	self.logger.save_state({'env': self.env}, None)

			# Perform PPO update!
			pi_l_new, v_l_new = self.update()

			# Log info about epoch
			self.logger.log_tabular('Epoch', epoch)
			self.logger.log_tabular('EpRet', with_min_and_max=True)
			self.logger.log_tabular('EpLen', average_only=True)
			self.logger.log_tabular('VVals', with_min_and_max=True)
			self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
			self.logger.log_tabular('LossPi', average_only=True)
			self.logger.log_tabular('LossV', average_only=True)
			self.logger.log_tabular('DeltaLossPi', average_only=True)
			self.logger.log_tabular('DeltaLossV', average_only=True)
			self.logger.log_tabular('Entropy', average_only=True)
			self.logger.log_tabular('KL', average_only=True)
			self.logger.log_tabular('ClipFrac', average_only=True)
			self.logger.log_tabular('StopIter', average_only=True)
			self.logger.log_tabular('Time', time.time() - start_time)
			self.logger.dump_tabular()

			# Update summaries per epoch
			summ = self.sess.run(self.loss_summaries, feed_dict={self.pi_loss_ph: pi_l_new, self.v_loss_ph: v_l_new})
			self.summ_writer.add_summary(summ, epoch)

			s0 = self.env.reset()
			a0 = self.predict(s0)
			s1, r, d, _ = self.env.step(a0)
			s1_model, r_model = self.env.testActualModel(a0)

			accuracy = np.sqrt(np.mean(np.power(np.ones(self.obs_dim[0]) - s1, 2)))

			summ = self.sess.run(self.accuracy_summary, feed_dict={self.accuracy_ph: accuracy})
			self.summ_writer.add_summary(summ, epoch)

			self.logger.log(f'Randam initial state:  reward = {r}, {s0}', 'magenta')
			self.logger.log(f'Resulting state:       {s1}', 'magenta')
			self.logger.log(f'Resulting state through actual model:  reward = {r_model}, {s1_model}', 'gray')

		# s1_list.append(s1)

		# ax.clear()
		# obs_dim = self.obs_dim[0]
		# ax.plot(range(obs_dim), np.ones(obs_dim), 'k', label="Reference")
		# for i, s in enumerate(s1_list):
		# 	if i < len(s1_list)-1:
		# 		ax.plot(range(obs_dim), s, '--')
		# 	else:
		# 		ax.plot(range(obs_dim), s, label=f"Epoch {epoch} prediction")

		# ax.legend(loc="upper right")
		# plt.pause(0.01)

	def predict(self, o):
		a, _, _ = self.sess.run(self.get_action_ops, feed_dict={self.x_ph: o.reshape(1, -1)})

		# self.logger.log_tabular('Predicted Action', a)

		return a[0]


if __name__ == '__main__':
	try:
		ppo = PPO(env_fn=MORsimpleEnv, epochs=100, steps_per_epoch=100, ac_kwargs={'hidden_sizes': (10,)}, logger_kwargs={'output_dir':'TestingSVDMOR', 'exp_name':'PPO-0'})
		ppo.run()
	except KeyboardInterrupt:
		try:
			os.makedirs('agents')
		except FileExistsError:
			pass
		import pickle
		pickle.dump(ppo, f"agents/PPO_{len(os.listdir('agents')) + 1}")


	ppo.sess.close()
