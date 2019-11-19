import os, sys

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
sys.path.append(os.path.join(dirname, "../../rl_ofc/src/gym-lhc/gym_lhc/envs"))
import numpy as np
import matplotlib.pyplot as plt

from lhc_env import LHCData

if __name__ == '__main__':
	STEPS = 1000

	lhcData = LHCData()
	lhcData.loadDataFromOfsuFile()
	rm = lhcData.getRM()

	U, Sigma, Vt = np.linalg.svd(rm)

	n_evals = 1

	fig, ax = plt.subplots()
	fig.suptitle("Response Matrix - Model Order Reduction - SVD")
	ax2 = plt.twinx(ax)
	plt.ion()
	plt.show(block=False)

	rmse = []
	rmse_std = []
	accuracy = []
	accuracy_std = []
	n_evals_list = []
	for n_evals in np.arange(550, 99, -1):
		n_evals_list.append(n_evals)

		U_trunc = U[:, :n_evals]
		Sigma_trunc = Sigma[:n_evals]
		V = Vt.T
		V_trunc = V[:, :n_evals]
		Vt_trunc = V_trunc.T

		S = np.diag(Sigma_trunc)

		B = U_trunc.dot(S)

		STEPS = 400

		rmse_temp = []
		accuracy_temp = []
		for _ in range(STEPS):
			input_vector = np.random.uniform(-1, 1, rm.shape[0])

			from_rm = np.linalg.multi_dot([input_vector, rm])
			temp = input_vector.dot(B)
			from_reduced_rm = temp.dot(Vt_trunc)

			rmse_temp.append(np.sqrt(np.mean(np.power(np.subtract(from_rm, from_reduced_rm), 2))))

			rms = np.sqrt(np.mean(np.power(from_rm, 2)))
			acc = ((rms - rmse) / rms) * 100
			accuracy_temp.append(acc)

		rmse.append(np.mean(rmse_temp))
		rmse_std.append(np.std(rmse_temp)/2)
		accuracy.append(np.mean(accuracy_temp))
		accuracy_std.append(np.std(accuracy_temp)/2)

		ax.clear()
		ax2.clear()

		l, = ax.plot(n_evals_list, rmse, 'b-')
		ax.fill_between(n_evals_list, np.subtract(rmse, rmse_std), np.add(rmse, rmse_std), facecolor='#9999ff', alpha=0.5)
		ax.set_xlabel("Number of eigen values used")
		ax.set_ylabel("RMSE to original RM")

		l2, = ax2.plot(n_evals_list, accuracy,'r-')
		ax2.fill_between(n_evals_list, np.subtract(accuracy, accuracy_std), np.add(accuracy, accuracy_std), facecolor='#ff9999', alpha=0.5)
		ax2.set_ylabel("Accuracy (%)")

		ax.yaxis.label.set_color(l.get_color())
		ax2.yaxis.label.set_color(l2.get_color())

		ax.spines['left'].set_edgecolor(l.get_color())
		ax2.spines['right'].set_edgecolor(l2.get_color())

		ax.tick_params(axis='y', colors=l.get_color())
		ax2.tick_params(axis='y', colors=l2.get_color())

		plt.grid()

		plt.pause(0.01)

	plt.show(block=True)
