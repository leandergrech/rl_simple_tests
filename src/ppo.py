import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('simpleEnv:simpleEnv-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
rewards=[]
random_env = gym.make('simpleEnv:simpleEnv-v0')
for i in range(1000):
	action, _states = model.predict(obs)
	obs, reward, dones, info = env.step(action)
	rewards.append(reward)
	env.render(rewards=rewards)