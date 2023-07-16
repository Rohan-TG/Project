import warnings
import scipy.stats
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import anomaly_remover, log_make_train, log_make_test, range_setter, delogger
import scipy.stats



df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]

df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))


df_test = anomaly_remover(dfa = df_test)

# def custom_loss(y_pred, y_true):
# 	num = y_pred + 1
# 	den = y_true + 1
#
# 	ln = np.log(y_pred + 1)
#
# 	grad = (num/den) - ln
# 	hess = np.repeat(2, y_true.shape[0])
#
# 	return grad, hess

al = range_setter(la=30, ua=215, df=df)

magic_numbers = [2, 8, 20, 28, 50, 82, 126]

doubly_magic = []
n_magic = []
p_magic = []
all_magic = []

for nuc in al:
	if nuc[0] in magic_numbers and (nuc[1] - nuc[0]) in magic_numbers:
		# print(f"Double: {nuc} - {periodictable.elements[nuc[0]]}-{nuc[1]}")
		doubly_magic.append(nuc)
		all_magic.append(nuc)
	elif nuc[0] in magic_numbers and (nuc[1] - nuc[0]) not in magic_numbers:
		# print(f"Protonic: {nuc} - {periodictable.elements[nuc[0]]}-{nuc[1]}")
		p_magic.append(nuc)
		all_magic.append(nuc)
	elif nuc[0] not in magic_numbers and (nuc[1] - nuc[0]) in magic_numbers:
		# print(f"Neutronic: {nuc} - {periodictable.elements[nuc[0]]}-{nuc[1]}")
		n_magic.append(nuc)
		all_magic.append(nuc)



log_reduction_var = 0.00001




n_evaluations = 30
datapoint_matrix = []

for i in range(n_evaluations):
	print(f"Run {i+1}/{n_evaluations}")

	validation_nuclides = [[82,208]]
	validation_set_size = 20  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("\nTest nuclide selection complete")

	time1 = time.time()

	X_train, y_train = log_make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215,
									  log_reduction_variable=log_reduction_var) # make training matrix

	X_test, y_test = log_make_test(validation_nuclides, df=df_test, log_reduction_variable=log_reduction_var)

	print("Data prep done")

	model_seed = random.randint(a=1, b=1000)

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.01,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=model_seed,)


	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions

	if i == 0:
		for k, pred in enumerate(predictions):
			if [X_test[k,0], X_test[k,1]] == validation_nuclides[0]:
				datapoint_matrix.append([np.exp(pred) - log_reduction_var])
	else:
		valid_predictions = []
		for k, pred in enumerate(predictions):
			if [X_test[k, 0], X_test[k, 1]] == validation_nuclides[0]:
				valid_predictions.append(np.exp(pred) - log_reduction_var)
		for m, prediction in zip(datapoint_matrix, valid_predictions):
			m.append(prediction)

	# shap_val_gpu = model.predict(X_test, pred_interactions=True)

	# Form arrays for plots below
	XS_plotmatrix = []
	E_plotmatrix = []
	P_plotmatrix = []
	for nuclide in validation_nuclides:
		dummy_test_XS = []
		dummy_test_E = []
		dummy_predictions = []
		for i, row in enumerate(X_test):
			if [row[0], row[1]] == nuclide:
				dummy_test_XS.append(np.exp(y_test[i]) - log_reduction_var)
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(np.exp(predictions[i]) - log_reduction_var)

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i] # validation nuclide
		# plt.plot(erg, pred_xs, label='predictions')
		# plt.plot(erg, true_xs, label='data')
		# plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
		# plt.legend()
		# plt.grid()
		# plt.ylabel('XS / b')
		# plt.xlabel('Energy / MeV')
		# plt.show()

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		# print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
		# time.sleep(1.2)

	exp_true_xs = [np.exp(y) -log_reduction_var for y in y_test]
	exp_pred_xs = [np.exp(xs)- log_reduction_var for xs in predictions]

	print(f"MSE: {mean_squared_error(exp_true_xs, exp_pred_xs, squared=False)}") # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs):0.5f}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')




XS_plot = []
E_plot = []

for i, row in enumerate(X_test):
	if [row[0], row[1]] == validation_nuclides[0]:
		XS_plot.append(np.exp(y_test[i]) - log_reduction_var)
		E_plot.append(row[4]) # Energy values are in 5th row

# print(datapoint_matrix)

datapoint_means = []
datapoint_upper_interval = []
datapoint_lower_interval = []

for point in datapoint_matrix:
	# print(point)
	mu = np.mean(point)
	sigma = np.std(point)
	interval_low, interval_high = scipy.stats.t.interval(0.05, loc=mu, scale=sigma, df=(len(point) - 1))
	datapoint_means.append(mu)
	datapoint_upper_interval.append(interval_high)
	datapoint_lower_interval.append(interval_low)


lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point-low)
	upper_bound.append(point+up)


# print(datapoint_means)
plt.plot(E_plot, datapoint_means, label = 'Prediction')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII')
plt.fill_between(E_plot, lower_bound, upper_bound, alpha=0.4, label='95% CF')
plt.legend()
plt.show()
