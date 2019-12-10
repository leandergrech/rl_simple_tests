import gym
import numpy as np

env = gym.make('simpleEnv:deltaSimpleEnv-v0', rm_use_last=False)

rm = env.response_matrix
pinv = np.linalg.inv(rm)
s = env.reset()

rewards = []
for i in range(1000):
	a = np.dot(pinv, s)

	s,r, _, _ = env.step(-a)

	rewards.append(r)
	env.render(rewards)

	print(r)
