import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matrix_functions import range_setter, r2_standardiser, General_plotter, exclusion_func
from propagator_functions import make_train_sampler, make_test_sampler
from sklearn.metrics import r2_score, mean_squared_error
import time
import periodictable
from datetime import timedelta
import tqdm
import random
runtime = time.time()

ENDFBVIII = pd.read_csv('ENDFBVIII_100keV_all_uncertainties.csv')
ENDFBVIII.index = range(len(ENDFBVIII))
ENDFB_nuclides = range_setter(df=ENDFBVIII, la=30, ua= 210)

TENDL = pd.read_csv("TENDL_2021_MT_16_all_u.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)

exc = exclusion_func() # 10 sigma with handpicked additions


target_nuclides = [[52,118]]
target_nuclide = target_nuclides[0]
n_evaluations = 5


def diff(target):

	match_element = []
	for nuclide in ENDFB_nuclides:
		if nuclide[0] == target[0]:
			match_element.append(nuclide)
	nuc_differences = []
	for match_nuclide in match_element:
		difference = match_nuclide[1] - target[1]
		nuc_differences.append(difference)

	if len(nuc_differences) > 0:
		if nuc_differences[0] > 0:
			min_difference = min(nuc_differences)
		elif nuc_differences[0] < 0:
			min_difference = max(nuc_differences)

	return (-1 * min_difference)








validation_set_size = 1

gate = 0.05

all_consensus_rmses = []

jendl_r2s=[]
jendl_rmses=[]

tendl_rmses=[]
tendl_r2s=[]

runs_r2_array = []

datapoint_matrix = []

print('Data loaded...')




tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])

for i in tqdm.tqdm(range(n_evaluations)):

	single_run_concat_libs = []
	single_run_concat_preds = []

	model_seed = random.randint(a=1, b=10000)
	X_train, y_train = make_train_sampler(df=ENDFBVIII, la=30, ua=210,
										validation_nuclides=target_nuclides,
										exclusions=exc,
										use_tqdm=False)

	X_test, y_test = make_test_sampler(nuclides=target_nuclides,
									   df=TENDL,
									   use_tqdm=False)

	model = xg.XGBRegressor(n_estimators=950,  # define regressor
							learning_rate=0.008,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=42, )

	model.fit(X_train, y_train)

	predictions = model.predict(X_test) # XS predictions
	predictions_ReLU = []
	for pred in predictions:
		if pred > (gate * max(predictions)):
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

	if target_nuclide in JENDL_nuclides:
		jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
		nativeerg, unused = General_plotter(df=TENDL, nuclides=[target_nuclide])
		interpolation_function_loop = scipy.interpolate.interp1d(nativeerg, y=predictions, fill_value='extrapolate')

		jendlxs_loop_interpolated = interpolation_function_loop(jendlerg)

		d1, d2, jendl_r2 = r2_standardiser(library_xs=jendlxs, predicted_xs=jendlxs_loop_interpolated)

		jendl_loop_rmse = mean_squared_error(d1,d2) ** 0.5

		jendl_r2s.append(jendl_r2)
		jendl_rmses.append(jendl_loop_rmse)

		for l, p in zip(jendlxs,jendlxs_loop_interpolated):
			single_run_concat_libs.append(l)
			single_run_concat_preds.append(p)

			all_libs.append(l)
			all_preds.append(p)

	gated_predictions, gated_tendl, pred_tendl_r2 = r2_standardiser(library_xs=tendlxs, predicted_xs=predictions)
	tendl_r2s.append(pred_tendl_r2)
	rmsetendl = mean_squared_error(gated_predictions, gated_tendl) ** 0.5
	tendl_rmses.append(rmsetendl)
	for x, y in zip(gated_predictions, gated_tendl):
		all_libs.append(y)
		all_preds.append(x)

		single_run_concat_libs.append(y)
		single_run_concat_preds.append(x)

	consensus_run_rmse = mean_squared_error(single_run_concat_libs, single_run_concat_preds) **0.5
	all_consensus_rmses.append(consensus_run_rmse)


consensus = r2_score(all_libs, all_preds)



runs_r2_array.append(consensus)
print(f"Consensus R2: {r2_score(all_libs, all_preds):0.5f}")

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


cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
endfberg, endfbxs = General_plotter(df=ENDFBVIII, nuclides=[target_nuclide])


interpolation_function = scipy.interpolate.interp1d(E_plot, y=datapoint_means, fill_value='extrapolate')

if target_nuclide in JENDL_nuclides:

	jendlxs_interpolated = interpolation_function(jendlerg)

	d1, d2, jendl_r2 = r2_standardiser(library_xs=jendlxs, predicted_xs=jendlxs_interpolated)

	print(f"R2 w.r.t. JENDL-5: {jendl_r2}")
d3,d4, tendlr2 = r2_standardiser(library_xs=tendlxs, predicted_xs=datapoint_means)
print(f"R2 w.r.t. TENDL-2021: {tendlr2}")
#2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
if target_nuclide in ENDFB_nuclides:
	plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendlerg, tendlxs, label = 'TENDL-2021', color='dimgrey')
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
	print(f'JENDL-5 RMSE: {np.mean(jendl_rmses):0.3f} +/- {np.std(jendl_rmses):0.3f}')
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

print(f"Mass difference: {diff(target_nuclide)}")
print(f'TENDL-2021 RMSE: {np.mean(tendl_rmses):0.3f} +/- {np.std(tendl_rmses):0.3f}')
if target_nuclide in JENDL_nuclides:
	print(f'JENDL-5 RMSE: {np.mean(jendl_rmses):0.3f} +/- {np.std(jendl_rmses):0.3f}')

print(f'Consensus RMSE: {np.mean(all_consensus_rmses):0.3f} +/- {np.std(all_consensus_rmses):0.3f}')

print()
print(f"Runtime: {timedelta(seconds=final_runtime)}")