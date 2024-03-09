from matrix_functions import make_test, make_train, range_setter, General_plotter, r2_standardiser, dsigma_dE
import pandas as pd
import xgboost as xg
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
import shap
import periodictable
import tqdm
import numpy as np
import scipy
from datetime import timedelta

time1 = time.time()

ENDFB = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
TENDL = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JEFF = pd.read_csv('JEFF33_all_features.csv')
CENDL = pd.read_csv('CENDL32_all_features.csv')

ENDFB_nuclides = range_setter(df=ENDFB, la=30, ua=210)
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)

all_libraries = pd.read_csv('MT16_all_libraries_mk1.csv')
all_libraries.index = range(len(all_libraries))

n_evaluations = 3
datapoint_matrix = []
target_nuclide = [28,60]

jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])
endfberg, endfbxs = General_plotter(df=ENDFB, nuclides=[target_nuclide])

runs_r2_array = []

for i in tqdm.tqdm(range(n_evaluations)):
	print(f"\nRun {i+1}/{n_evaluations}")


	validation_nuclides = [target_nuclide]
	validation_set_size = 1  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(ENDFB_nuclides)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("\nTest nuclide selection complete")

	X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides,
								  la=30, ua=210)

	X_test, y_test = make_test(nuclides=validation_nuclides, df=ENDFB)


	print("Matrices formed...")

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.008,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=model_seed,)

	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions
	predictions_ReLU = []
	for pred in predictions:
		if pred >= 0.003:
			predictions_ReLU.append(pred)
		else:
			predictions_ReLU.append(0)

	predictions = predictions_ReLU

	if i == 0:
		for k, pred in enumerate(predictions):
			if [X_test[k,0], X_test[k,1]] == validation_nuclides[0]:
				datapoint_matrix.append([pred])
	else:
		valid_predictions = []
		for k, pred in enumerate(predictions):
			if [X_test[k, 0], X_test[k, 1]] == validation_nuclides[0]:
				valid_predictions.append(pred)
		for m, prediction in zip(datapoint_matrix, valid_predictions):
			m.append(prediction)

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
				dummy_test_XS.append(y_test[i])
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	all_preds = []
	all_libs = []

	pred_endfb_r2 = r2_score(y_true=XS_plotmatrix[0], y_pred=P_plotmatrix[0])
	for x, y in zip(XS_plotmatrix[0], P_plotmatrix[0]):
		all_libs.append(y)
		all_preds.append(x)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

	if target_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=pred_cendl)
		for x, y in zip(pred_cendl, cendl_xs):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f}")

	if target_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=pred_jendl)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")
		for x, y in zip(pred_jendl, jendl_xs):
			all_libs.append(y)
			all_preds.append(x)

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f}")

	if target_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=pred_tendl)
		for x, y in zip(pred_tendl, tendl_xs):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f}")
	consensus = r2_score(all_libs, all_preds)

	runs_r2_array.append(consensus)
	print(f"Consensus R2: {r2_score(all_libs, all_preds):0.5f}")

	exp_true_xs = [y for y in y_test]
	exp_pred_xs = [xs for xs in predictions]

	print(f'completed in {time.time() - time1:0.1f} s')

XS_plot = []
E_plot = []

for i, row in enumerate(X_test):
	if [row[0], row[1]] == target_nuclide:
		XS_plot.append(y_test[i])
		E_plot.append(row[4]) # Energy values are in 5th row

datapoint_means = []
datapoint_upper_interval = []
datapoint_lower_interval = []

d_l_1sigma = []
d_u_1sigma = []

for point in datapoint_matrix:

	mu = np.mean(point)
	sigma = np.std(point)
	interval_low, interval_high = scipy.stats.t.interval(0.95, loc=mu, scale=sigma, df=(len(point) - 1))
	# il_1sigma, ih_1sigma = scipy.stats.t.interval(0.68, loc=mu, scale=sigma, df=(len(point) - 1))
	datapoint_means.append(mu)
	datapoint_upper_interval.append(interval_high)
	datapoint_lower_interval.append(interval_low)

lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point-low)
	upper_bound.append(point+up)

print(f"Turning points: {dsigma_dE(XS=datapoint_means)}")

#2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendlerg, tendl_xs, label = 'TENDL-2021', color='dimgrey')
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
if target_nuclide in CENDL_nuclides:
	plt.plot(cendlerg, cendlxs, '--', label = 'CENDL-3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')

plt.grid()
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.legend(loc='upper left')
plt.show()

mu_r2 = np.mean(runs_r2_array)
sigma_r2 = np.std(runs_r2_array)
interval_low_r2, interval_high_r2 = scipy.stats.t.interval(0.95, loc=mu_r2, scale=sigma_r2, df=(len(runs_r2_array) - 1))
print(f"\nConsensus: {mu_r2:0.5f} +/- {interval_high_r2-interval_low_r2:0.5f}")

final_runtime = time.time() - time1

print()
print(f"Runtime: {timedelta(seconds=final_runtime)}")