import numpy as np
import matplotlib.pyplot as plt
import gym
from simpleEnv.envs import MORsimpleEnv

if __name__ == '__main__':
	SIZE = 550
	ref_env = gym.make('simpleEnv:simpleEnv-v0', rm_use_last=False, rm_kwargs=dict(rm_size=SIZE))
	sample_action = ref_env.action_space.sample()
	ref_state = ref_env.step(sample_action)[0]

	fig, [ax, ax2] = plt.subplots(2)

	rms = []
	rms_i = []

	for i in range(SIZE):
		if i % 50 != 0 or i==0:
			continue
		env = gym.make('simpleEnv:MORsimpleEnv-v0', rm_use_last=True, rm_kwargs=dict(n_evals=i))
		print(f"Env {i}")
		eigen_action = np.dot(env.action_decoder, sample_action)
		s, _, _, _ = env.step(eigen_action)
		r = np.mean(np.sqrt(np.power(np.subtract(s, ref_state),2)))
		ax.plot(s, label=f"Action size = {i}")
		rms.append(r)
		rms_i.append(i)

	ax.plot(ref_env.step(sample_action)[0], color='k', linewidth=2, label="Reference")
	ax.legend(loc='best')

	ax2.set_title("RMSE from ref vs action size")
	ax2.set_xlabel("Action size")
	ax2.set_ylabel("RMSE")
	ax2.plot(rms_i, rms)

	plt.grid()
	plt.show(block=True)
