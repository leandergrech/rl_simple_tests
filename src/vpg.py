import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../spinup"))

from functools import partial

import spinup
from simple_env import simpleEnv
from datetime import datetime as dt
import time

if __name__ == '__main__':
	exp_string = f"vpg-{dt.now().strftime('%d%m%y-%H%M')}"
	env_fn = partial(simpleEnv, rm_size=5)

	spinup.vpg(env_fn=env_fn, seed=int(time.time()), steps_per_epoch=1000, epochs=100, gamma=0.99, pi_lr=5e-4, vf_lr=1e-3,
			   train_v_iters=80, lam=0.96, max_ep_len=1000, logger_kwargs={"output_dir": os.path.join("results", exp_string), "exp_name": exp_string}, save_freq=15)
