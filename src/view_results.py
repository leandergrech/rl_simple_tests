import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import numpy as np

env = gym.make('simpleEnv:simpleEnv-v0')
vec_env = DummyVecEnv([lambda: env])

model =PPO2.load('simpleEnv-full5x5', vec_env, verbose=0, tensorboard_log='learning-ppo')

obs = env.reset()
rewards = []
for i in range(1000):
	env.render(rewards=rewards)

	if i == 20:
		env.reference_trajectory = np.random.normal(1,1, env.obs_dimension)

	action, _ = model.predict(obs)
	s, r, _,_ = env.step(action)
	rewards.append(r)
