import unittest
import gym
import numpy.testing as nptest


class compareEnvs(unittest.TestCase):
	def test_rm_size(self):
		size = 5
		env = gym.make('simpleEnv:simpleEnv-v0', rm_use_last=False, rm_kwargs=dict(rm_size=size))
		morenv = gym.make('simpleEnv:MORsimpleEnv-v0', rm_use_last=True, rm_kwargs=dict(n_evals=size))

		nptest.assert_array_equal(env.response_matrix.shape, morenv.response_matrix.shape)

if __name__ == '__main__':
    unittest.main()