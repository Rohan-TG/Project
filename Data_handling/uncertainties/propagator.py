import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matrix_functions import range_setter, r2_standardiser, General_plotter
from propagator_functions import training_sampler, make_test
from sklearn.metrics import r2_score, mean_squared_error
import time
import periodictable
from datetime import timedelta
import tqdm

runtime = time.time()

ENDFBVIII = pd.read_csv('ENDFBVIII_MT16_uncertainties.csv')
ENDFB_nuclides = range_setter(df=ENDFBVIII, la=0, ua= 210)

TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=210)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=210)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=210)

exc = [[22, 47], [65, 159], [66, 157], [38, 90], [61, 150],
	   [74, 185], [50, 125], [50, 124], [60, 149], [39, 90],
	   [64, 160], [38, 87], [39, 91], [63, 152], [52, 125],
	   [19, 40], [56, 139], [52, 126], [71, 175], [34, 79],
	   [70, 175], [50, 117], [23, 49], [63, 156], [57, 140],
	   [52, 128], [59, 142], [50, 118], [50, 123], [65, 161],
	   [52, 124], [38, 85], [51, 122], [19, 41], [54, 135],
	   [32, 75], [81, 205], [71, 176], [72, 175], [50, 122],
	   [51, 125], [53, 133], [34, 82], [41, 95], [46, 109],
	   [84, 209], [56, 140], [64, 159], [68, 167], [16, 35],
	   [18,38]] # 10 sigma with handpicked additions

target_nuclides = [[71,175]]
target_nuclide = target_nuclides[0]
n_evaluations = 100

runs_r2_array = []

datapoint_matrix = []

X_train, y_train = training_sampler(df=ENDFBVIII, LA=30, UA=210, sampled_uncertainties= ['unc_sn'],
									target_nuclides=target_nuclides, exclusions=exc)
print("Training matrix formed")

model = xg.XGBRegressor(n_estimators=900,  # define regressor
						learning_rate=0.008,
						max_depth=8,
						subsample=0.18236,
						max_leaves=0,
						seed=42, )

model.fit(X_train, y_train)
print("Training complete")

for i in tqdm.tqdm(range(n_evaluations)):
	print(f"\nRun {i + 1}/{n_evaluations}")

	time1 = time.time()


	X_test, y_test = make_test(nuclides=target_nuclides, df=ENDFBVIII)

	print('Test data generation complete')


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
			if [X_test[k,0], X_test[k,1]] == target_nuclides[0]:
				datapoint_matrix.append([pred])
	else:
		valid_predictions = []
		for k, pred in enumerate(predictions):
			if [X_test[k, 0], X_test[k, 1]] == target_nuclides[0]:
				valid_predictions.append(pred)
		for m, prediction in zip(datapoint_matrix, valid_predictions):
			m.append(prediction)

	XS_plotmatrix = []
	E_plotmatrix = []
	P_plotmatrix = []

	for nuclide in target_nuclides:
		dummy_test_XS = []
		dummy_test_E = []
		dummy_predictions = []
		for i, row in enumerate(X_test):
			if [row[0], row[1]] == nuclide:
				dummy_test_XS.append(y_test[i])
				dummy_test_E.append(row[4])  # Energy values are in 5th row
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

	# if target_nuclide in CENDL_nuclides:
	#
	# 	cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL)
	# 	pred_cendl = model.predict(cendl_test)
	#
	# 	pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=pred_cendl)
	# 	for x, y in zip(pred_cendl, cendl_xs):
	# 		all_libs.append(y)
	# 		all_preds.append(x)
	# 	print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f}")
	#
	# if target_nuclide in JENDL_nuclides:
	#
	# 	jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL)
	# 	pred_jendl = model.predict(jendl_test)
	#
	# 	pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=pred_jendl)
	# 	print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")
	# 	for x, y in zip(pred_jendl, jendl_xs):
	# 		all_libs.append(y)
	# 		all_preds.append(x)
	#
	# if target_nuclide in JEFF_nuclides:
	# 	jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF)
	#
	# 	pred_jeff = model.predict(jeff_test)
	#
	# 	pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff,
	# 																	library_xs=jeff_xs)
	# 	for x, y in zip(pred_jeff_gated, truncated_jeff):
	# 		all_libs.append(y)
	# 		all_preds.append(x)
	# 	print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f}")
	#
	# if target_nuclide in TENDL_nuclides:
	#
	# 	tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL)
	# 	pred_tendl = model.predict(tendl_test)
	#
	# 	pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
	# 	pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=pred_tendl)
	# 	for x, y in zip(pred_tendl, tendl_xs):
	# 		all_libs.append(y)
	# 		all_preds.append(x)
	# 	print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f}")
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

	# d_l_1sigma.append(il_1sigma)
	# d_u_1sigma.append(ih_1sigma)

lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point-low)
	upper_bound.append(point+up)

# print(f"Turning points: {dsigma_dE(XS=datapoint_means)}")

jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])
endfberg, endfbxs = General_plotter(df=ENDFBVIII, nuclides=[target_nuclide])

#2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendlerg, tendlxs, label = 'TENDL-2021', color='dimgrey')
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
if target_nuclide in CENDL_nuclides:
	plt.plot(cendlerg, cendlxs, '--', label = 'CENDL-3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
plt.grid()
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[target_nuclide[0]]}-{target_nuclide[1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.legend(loc='upper left')
plt.show()

final_runtime = time.time() - runtime



print()
print(f"Runtime: {timedelta(seconds=final_runtime)}")