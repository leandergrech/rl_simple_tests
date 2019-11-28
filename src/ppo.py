import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import argparse
from time import sleep

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('total_timesteps')
	args = parser.parse_args()

	env = gym.make('simpleEnv:simpleEnv-v0')
	print(f"Starting training on env having shape = {env.shape}")

	env = DummyVecEnv([lambda: env])
	model = PPO2(MlpPolicy, env, tensorboard_log='log', verbose=1)

	sleep(1)

	try:
		model.learn(total_timesteps=int(args.total_timesteps))
	except KeyboardInterrupt:
		pass


	model.save('simpleEnv-full550x550')

