import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../spinup"))

from functools import partial

import spinup
from simple_env import simpleEnv
from datetime import datetime as dt
import time
from spinup.utils.mpi_tools import mpi_fork

if __name__ == '__main__':
	exp_string = f"vpg-{dt.now().strftime('%d%m%y-%H%M')}"
	env_fn = partial(simpleEnv, rm_size=5)

	mpi_fork(4)

	spinup.vpg(env_fn=env_fn, seed=int(time.time()), steps_per_epoch=1000, epochs=250, gamma=0.99, pi_lr=0.1e-3, vf_lr=0.8e-3,
			   train_v_iters=80, lam=0.97, max_ep_len=1000, logger_kwargs={"output_dir": os.path.join("results", exp_string), "exp_name": exp_string}, save_freq=50)
