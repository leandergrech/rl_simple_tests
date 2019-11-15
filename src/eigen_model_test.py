import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm as progressbar

def pmat(mat):
	for i, row in enumerate(mat):
		if i != 0:
			print()
		for elem in row:
			print(elem, end='\t')

def pca():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	# load dataset into Pandas DataFrame
	df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

	features = ['sepal length', 'sepal width', 'petal length', 'petal width']

	# Separating out the features
	x = df.loc[:, features].values
	# Separating out the target
	y = df.loc[:, ['target']].values
	# Standardizing the features
	x = StandardScaler().fit_transform(x)

	print(df)
	print(x)

def svd_reduction(model, n_s):
	u, s, vh = np.linalg.svd(model)

	s_reducted = s.copy()

	s_reducted[n_s:] = 0.0

	return np.linalg.multi_dot([u, np.pad(np.diag(s_reducted), ((0, len(u)-len(s_reducted)),(0,0))), vh])

def eig_reduction(model, n_evals):
	evals, evecs = np.linalg.eig(model)

	assert n_evals <= model.shape[0]

	evals_reducted = evals.copy()

	# idx = np.argsort(evals_reducted)[::-1]
	# evals_reducted = evals_reducted[idx]
	# evecs = evecs[idx]

	evals_reducted[n_evals:] = 0.0

	return np.linalg.multi_dot([evecs, np.diag(evals_reducted), np.linalg.inv(evecs)])

def eval_model_all_eigenvalues(model, N_STEPS, decomposition_callable = eig_reduction):
	INPUT_SIZE = model.shape[0]
	MAX_EVALS = min(model.shape)

	n_evals_list = np.arange(MAX_EVALS, 0, -1, dtype=np.int)
	rmse = defaultdict(list)

	for n_evals in n_evals_list:
		if n_evals % 25 != 0 and n_evals != 1:
			continue
		print(f"Testing with {n_evals} eigenvalues")
		model_reducted = decomposition_callable(model, n_evals)
		for _ in range(N_STEPS):
			input = np.random.uniform(-1,1, INPUT_SIZE)
			from_model = np.dot(input, model)
			from_reducted_model = np.dot(input, model_reducted)

			error = np.sqrt(np.mean(np.power(np.subtract(from_model, from_reducted_model),2)))

			rmse[n_evals].append(error)

	return rmse

def eval_random_model_all_eigenvalues():
	SIZE = 10
	N_MODELS = 100
	N_STEPS = 500



	rmse = []
	n_evals_list = np.arange(SIZE, 0, -1, dtype=np.int)

	for n_model in progressbar(range(N_MODELS)):
		# a = np.random.normal(0.0, 1.0, (SIZE, SIZE))
		a = np.diag(np.random.normal(0, 1, SIZE))
		rmse.append(eval_model_all_eigenvalues(a, N_STEPS))



	mpl.style.use('seaborn')
	fig, ax = plt.subplots()

	stats = [[np.abs(model[n_eval]) for n_eval in model] for model in rmse]

	# print(rmse)
	# print(np.abs(list(rmse.values())))

	mean_error = np.mean(np.mean(stats, axis=2), axis=0)
	std_error = np.mean(np.std(stats, axis=2), axis=0)

	fig, ax = plt.subplots()
	ax.plot(n_evals_list, mean_error)
	ax.fill_between(n_evals_list, mean_error-std_error, mean_error+std_error, facecolor='#a9cce3')
	ax.set_xlabel("# Eigenvalues used")
	ax.set_ylabel("RMSE to original model")
	plt.show()

if __name__ == '__main__':
	import os, sys

	dirpath = os.path.dirname(os.path.realpath(__file__))
	sys.path.append(os.path.join(dirpath, "../../rl_ofc/src/gym-lhc/gym_lhc/envs"))
	from lhc_env import LHCData

	lhcData = LHCData()
	lhcData.loadDataFromOfsuFile()

	rm = lhcData.getRM()

	# rm = np.random.normal(0,1,(50,50))

	rmse = eval_model_all_eigenvalues(rm, 5, svd_reduction)
	stats = [rmse[n_evals] for n_evals in rmse]

	print(stats)
	mean = np.mean(stats, axis=1)
	print(mean)
	std = np.std(stats, axis=1)
	print(std)
 
	fig, ax = plt.subplots()
	x = np.linspace(min(rm.shape), 1, len(mean))
	ax.plot(x, mean)
	ax.fill_between(x, mean-std, mean+std, facecolor='#a9cce3')
	ax.set_xlabel("# Eigenvalues used")
	ax.set_ylabel("RMSE to original model")
	plt.show()




