import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import argparse


if __name__ == '__main__':
	model_name = 'simpleEnv-full550x550'	

	parser = argparse.ArgumentParser()
	parser.add_argument('total_timesteps')
	parser.add_argument('tb_log')
	args = parser.parse_args()

	env = gym.make('simpleEnv:simpleEnv-v0')
	print('\n\n')
	print(f"Starting training on env having shape = {env.response_matrix.shape}")
	print('\n\n')
	sleep(4)

	env = DummyVecEnv([lambda: env])
	model = PPO2(MlpPolicy, env, tensorboard_log=args.tb_log, verbose=1)

	try:
		model.learn(total_timesteps=int(args.total_timesteps))
	except KeyboardInterrupt:
		pass

	model.save(model_name)

