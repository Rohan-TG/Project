import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matrix_functions import range_setter, r2_standardiser, General_plotter, exclusion_func
from propagator_functions import fia_test_propagator, make_train_sampler
from sklearn.metrics import r2_score, mean_squared_error
import time
import periodictable
from datetime import timedelta
import tqdm

runtime = time.time()

ENDFBVIII = pd.read_csv('ENDFBVIII_100keV_all_uncertainties.csv')
ENDFB_nuclides = range_setter(df=ENDFBVIII, la=30, ua= 208)

TENDL = pd.read_csv("TENDL_2021_MT_16_all_u.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=208)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=208)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=208)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=208)

exc = exclusion_func()# 10 sigma with handpicked additions

target_nuclides = [[72,178]]
target_nuclide = target_nuclides[0]
n_evaluations = 8

runs_r2_array = []

datapoint_matrix = []



for i in tqdm.tqdm(range(n_evaluations)):

	endfb_r2s = []
	cendl_r2s = []
	tendl_r2s = []
	jeff_r2s = []
	jendl_r2s = []

	X_train, y_train = make_train_sampler(df=ENDFBVIII, exclusions=exc, validation_nuclides=target_nuclides,
								  la=30, ua=208,
										  use_tqdm=False)
	print("Data matrix formed, training...")

	model = xg.XGBRegressor(n_estimators=950,  # define regressor
							learning_rate=0.008,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=42, )

	model.fit(X_train, y_train)


	X_test, y_test = fia_test_propagator(nuclides=target_nuclides, df=ENDFBVIII)



	predictions = model.predict(X_test) # XS predictions
	predictions_ReLU = []
	for p in predictions:
		if p > (0.02 * max(predictions)):
			predictions_ReLU.append(p)
		else:
			predictions_ReLU.append(0.0)

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

	jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
	cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
	jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
	tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])
	endfberg, endfbxs = General_plotter(df=ENDFBVIII, nuclides=[target_nuclide])

	interpolation_function = scipy.interpolate.interp1d(endfberg, y=P_plotmatrix[0],
														fill_value='extrapolate')

	if target_nuclide in JENDL_nuclides:
		jendlxs_interpolated = interpolation_function(jendlerg)

		predjendlgated, d2, jendl_r2 = r2_standardiser(library_xs=jendlxs, predicted_xs=jendlxs_interpolated)
		jendl_r2s.append(jendl_r2)

		for x, y in zip(d2, predjendlgated):
			all_libs.append(x)
			all_preds.append(y)

	if target_nuclide in CENDL_nuclides:
		cendlxs_interpolated = interpolation_function(cendlerg)

		predcendlgated, truncatedcendl, cendl_r2 = r2_standardiser(library_xs=cendlxs,
																   predicted_xs=cendlxs_interpolated)
		cendl_r2s.append(cendl_r2)

		for x, y in zip(truncatedcendl, predcendlgated):
			all_libs.append(x)
			all_preds.append(y)

	if target_nuclide in JEFF_nuclides:
		jeffxs_interpolated = interpolation_function(jefferg)

		predjeffgated, d2, jeff_r2 = r2_standardiser(library_xs=jeffxs,
													 predicted_xs=jeffxs_interpolated)
		jeff_r2s.append(jeff_r2)
		for x, y in zip(d2, predjeffgated):
			all_libs.append(x)
			all_preds.append(y)

	tendlxs_interpolated = interpolation_function(tendlerg)
	predtendlgated, truncatedtendl, tendlr2 = r2_standardiser(library_xs=tendlxs, predicted_xs=tendlxs_interpolated)
	tendl_r2s.append(tendlr2)

	for x, y in zip(truncatedtendl, predtendlgated):
		all_libs.append(x)
		all_preds.append(y)
	consensus = r2_score(all_libs, all_preds)

	runs_r2_array.append(consensus)

	exp_true_xs = [y for y in y_test]
	exp_pred_xs = [xs for xs in predictions]


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
