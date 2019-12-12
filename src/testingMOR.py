import gym

env = gym.make('simpleEnv:simpleEnv-v0')
morenv = gym.make('simpleEnv:MORsimpleEnv-v0')

print(env.reset())
print(morenv.reset())

