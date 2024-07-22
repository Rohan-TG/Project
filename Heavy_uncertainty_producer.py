import datetime
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
from matrix_functions import dsigma_dE, make_train, make_test,\
	range_setter, General_plotter, r2_standardiser, exclusion_func
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df = df[df.Z != 11]

df.index = range(len(df))


al = range_setter(la=220, ua=260, df=df)

TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=260)


JEFF33 = pd.read_csv('JEFF33_all_features.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=260)


JENDL5 = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=260)


CENDL32 = pd.read_csv('CENDL32_all_features.csv')
CENDL32.index = range(len(CENDL32))
CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=260)

exc = exclusion_func()

n_evaluations = 100
datapoint_matrix = []
target_nuclide = [98,252]

jendlerg, jendlxs = General_plotter(df=JENDL5, nuclides=[target_nuclide])
cendlerg, cendlxs = General_plotter(df=CENDL32, nuclides=[target_nuclide])
jefferg, jeffxs = General_plotter(df=JEFF33, nuclides=[target_nuclide])
tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])
endfberg, endfbxs = General_plotter(df=df, nuclides=[target_nuclide])

runs_r2_array = []
runs_rmse_array = []

cendl_r2s = []
jeff_r2s = []
jendl_r2s = []
tendl_r2s = []
endfb_r2s = []

for i in tqdm.tqdm(range(n_evaluations)):
	# print(f"\nRun {i+1}/{n_evaluations}")

	validation_nuclides = [target_nuclide]
	validation_set_size = 1  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("\nTest nuclide selection complete")

	time1 = time.time()

	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides,
								  exclusions=exc,
								  la=220, ua=260,) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=df,)

	print("Training...")

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	# model = xg.XGBRegressor(n_estimators=950,
	# 						learning_rate=0.008,
	# 						max_depth=8,
	# 						subsample=0.18236,
	# 						max_leaves=0,
	# 						seed=model_seed,)


	model = xg.XGBRegressor(n_estimators=500,
							learning_rate=0.01,
							max_depth=8,
							subsample=0.2,
							max_leaves=0,
							seed=model_seed,)

	model.fit(X_train, y_train)

	print("Training complete")

	predictions_ReLU = []

	for n in validation_nuclides:

		temp_x, temp_y = make_test(nuclides=[n], df=df)
		initial_predictions = model.predict(temp_x)

		for p in initial_predictions:
			if p >= (0.02 * max(initial_predictions)):
				predictions_ReLU.append(p)
			else:
				predictions_ReLU.append(0.0)


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


	d1, d2, pred_endfb_r2 = r2_standardiser(library_xs=XS_plotmatrix[0],predicted_xs=P_plotmatrix[0])
	endfb_r2s.append(pred_endfb_r2)
	for x, y in zip(d1, d2):
		all_libs.append(y)
		all_preds.append(x)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

	if target_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL32)
		pred_cendl = model.predict(cendl_test)

		d1, d2, pred_cendl_r2 = r2_standardiser(library_xs=cendl_xs, predicted_xs=pred_cendl)
		cendl_r2s.append(pred_cendl_r2)
		for x, y in zip(d1, d2):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL-3.2 R2: {pred_cendl_r2:0.5f}")

	if target_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL5)
		pred_jendl = model.predict(jendl_test)

		d1,d2,pred_jendl_r2 = r2_standardiser(library_xs=jendl_xs, predicted_xs=pred_jendl)
		jendl_r2s.append(pred_jendl_r2)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")
		for x, y in zip(d1, d2):
			all_libs.append(y)
			all_preds.append(x)

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF33)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(predicted_xs=pred_jeff, library_xs=jeff_xs)
		jeff_r2s.append(pred_jeff_r2)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF-3.3 R2: {pred_jeff_r2:0.5f}")

	if target_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL)
		pred_tendl = model.predict(tendl_test)

		# pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		d1, d2, pred_tendl_r2 = r2_standardiser(library_xs=tendl_xs, predicted_xs=pred_tendl)
		tendl_r2s.append(pred_tendl_r2)
		for x, y in zip(d1, d2):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - TENDL-2021 R2: {pred_tendl_r2:0.5f}")
	consensus = r2_score(all_libs, all_preds)
	consensus_rmse = mean_squared_error(all_libs, all_preds) ** 0.5
	runs_r2_array.append(consensus)
	runs_rmse_array.append(consensus_rmse)

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

	# d_l_1sigma.append(il_1sigma)
	# d_u_1sigma.append(ih_1sigma)

lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point-low)
	upper_bound.append(point+up)










print(f"Turning points: {dsigma_dE(XS=datapoint_means)}")

#2sigma CF
plt.figure()
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
print(f"ENDF/B-VIII: {np.mean(endfb_r2s)} +- {np.std(endfb_r2s)}")
plt.plot(tendlerg, tendl_xs, label = 'TENDL-2021', color='dimgrey')
print(f"TENDL-2021: {np.mean(tendl_r2s):0.3f} +- {np.std(tendl_r2s):0.3f}")
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
	print(f"JEFF-3.3: {np.mean(jeff_r2s):0.3f} +- {np.std(jeff_r2s):0.3f}")
if target_nuclide in JENDL_nuclides:
	print(f"JENDL-5: {np.mean(jendl_r2s):0.3f} +- {np.std(jendl_r2s):0.3f}")
	plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
if target_nuclide in CENDL_nuclides:
	print(f"CENDL-3.2: {np.mean(cendl_r2s):0.3f} +- {np.std(cendl_r2s):0.3f}")
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

print(f'Consensus RMSE: {np.mean(runs_rmse_array):0.3f} +/- {np.std(runs_rmse_array):0.3f}')

final_runtime = time.time() - runtime



print()
print(f"Runtime: {timedelta(seconds=final_runtime)}")
