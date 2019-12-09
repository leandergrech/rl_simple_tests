import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import *

import argparse
from time import sleep
from functools import partial

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('total_timesteps')
	args = parser.parse_args()

	env = gym.make('simpleEnv:simpleEnv-v0')
	print('\n\n')
	print(f"Starting training on env having shape = {env.response_matrix.shape}")
	print('\n\n')

	env = DummyVecEnv([lambda: env])

	layers = [350, 350]
	net_arch = [dict(vf=layers, pi=layers)]
	# myMlpPolicy = partial(MlpPolicy, net_arch=net_arch)


	# model = PPO2(MlpPolicy, env, policy_kwargs={"net_arch":net_arch}, gamma=0.99, n_steps=500, tensorboard_log='log', verbose=1)
	model = SAC('MlpPolicy', env, tensorboard_log='log', verbose=1)

	sleep(1)

	try:
		model.learn(total_timesteps=int(args.total_timesteps))
	except KeyboardInterrupt:
		pass


	model.save('simpleEnv-full550x550-mlp600x600')

