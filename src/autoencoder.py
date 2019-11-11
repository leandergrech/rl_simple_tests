import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath("/home/lgrech/phd-code/rl_ofc/src/gym-lhc/gym_lhc/envs"))

from simple_env import simpleEnv
from lhc_env import LHCData

import numpy as np
import matplotlib.pyplot as plt

lhcData = LHCData()
lhcData.loadDataFromOfsuFile()
rm = lhcData.getRM()
act_dim = rm.shape[0]

env = simpleEnv(rm_size=act_dim)
s0 = env.reset()

a0 = np.random.normal(1.5, 0.2, env.act_dimension)

s = []

fig, ax = plt.subplots()
plt.show(block=False)
plt.ion()

a=a0
scale = 0.01
index_to_change = -150
for i in range(100):
	a = np.random.normal(1.5, 1, env.act_dimension)
	s = np.dot(a, rm)
	ax.plot(s, label=i)
	plt.pause(0.5)

input()



