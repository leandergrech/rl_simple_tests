import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('total_timesteps')
	args = parser.parse_args()

	env = gym.make('simpleEnv:simpleEnv-v0')
	env = DummyVecEnv([lambda: env])

	model = PPO2(MlpPolicy, env, verbose=1)
	model.learn(total_timesteps=int(args.total_timesteps))

	model.save('simpleEnv-full5x5')

