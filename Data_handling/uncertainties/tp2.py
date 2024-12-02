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
ENDFB_nuclides = range_setter(df=ENDFBVIII, la=0, ua= 208)

TENDL = pd.read_csv("TENDL_2021_MT_16_all_u.csv")
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
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=208)

class Colours:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

main_bar_format = '{l_bar}%s{bar}%s{r_bar}' % (Colours.BLUE, Colours.BLUE)

exc = exclusion_func() # 10 sigma with handpicked additions

nuclide_queue = [[30,64]]

for q_num in tqdm.tqdm(nuclide_queue, total=len(nuclide_queue), bar_format=main_bar_format):

	target_nuclide = q_num
	n_evaluations = 100

	jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
	cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
	jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
	tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])
	endfberg, endfbxs = General_plotter(df=ENDFBVIII, nuclides=[target_nuclide])


	validation_set_size = 20

	nuclide_thresholds = []


	runs_r2_array = []
	runs_rmse_array = []

	datapoint_matrix = []

	print('Data loaded...')

	endfb_r2s = []
	cendl_r2s = []
	tendl_r2s = []
	jeff_r2s = []
	jendl_r2s = []

	endfb_rmses = []
	cendl_rmses = []
	tendl_rmses = []
	jeff_rmses = []
	jendl_rmses = []

	second_bar_format = '{l_bar}%s{bar}%s{r_bar}' % (Colours.RED, Colours.RED)

	for i in tqdm.tqdm(range(n_evaluations), bar_format=second_bar_format):
		# print(f"\nRun {i + 1}/{n_evaluations}")
		model_seed = random.randint(a=1, b=10000)
		target_nuclides = [target_nuclide]

		while len(target_nuclides) < validation_set_size:  # up to 25 nuclides
			choice = random.choice(ENDFB_nuclides)  # randomly select nuclide from list of all nuclides in ENDF/B-VIII
			if choice not in target_nuclides:
				target_nuclides.append(choice)
		# print("Test nuclide selection complete")


		time1 = time.time()
		X_test, y_test = make_test_sampler(nuclides=target_nuclides, df=ENDFBVIII, use_tqdm=False)
		# print('Test data generation complete...')

		X_train, y_train = make_train_sampler(df=ENDFBVIII, la=30, ua=208,
											  validation_nuclides=target_nuclides,
											  exclusions=exc,
											  use_tqdm=False)
		# print("Training...")

		model = xg.XGBRegressor(n_estimators=950,  # define regressor
								learning_rate=0.008,
								max_depth=8,
								subsample=0.18236,
								max_leaves=0,
								seed=model_seed,)

		model.fit(X_train, y_train)
		# print("Training complete")

		predictions_ReLU = []

		for n in target_nuclides:
			tempx, tempy = make_test_sampler(nuclides=[n], df=ENDFBVIII, use_tqdm=False)
			initial_predictions = model.predict(tempx)

			for p in initial_predictions:
				if p > (0.02 * max(initial_predictions)):
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

		threshold_values = []

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

		for xs, energy in zip(P_plotmatrix[0], E_plotmatrix[0]):
			if xs > 0:
				predicted_threshold = energy
				break

		endfbgated, dee, pred_endfb_r2 = r2_standardiser(library_xs=XS_plotmatrix[0], predicted_xs=P_plotmatrix[0])
		endfb_r2s.append(r2_score(endfbgated, dee))
		for x, y in zip(dee, endfbgated):
			all_libs.append(x)
			all_preds.append(y)
		# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")
		for o, p in zip(endfberg, endfbxs):
			if p > 0:
				threshold_values.append(o)
				break
		interpolation_function = scipy.interpolate.interp1d(endfberg, y=P_plotmatrix[0],
															fill_value='extrapolate')

		if target_nuclide in JENDL_nuclides:
			jendlxs_interpolated = interpolation_function(jendlerg)

			predjendlgated, gated_jendl_xs, jendl_r2 = r2_standardiser(library_xs=jendlxs, predicted_xs=jendlxs_interpolated)
			jendl_r2s.append(jendl_r2)
			jendl_rmse = mean_squared_error(predjendlgated, gated_jendl_xs) **0.5
			jendl_rmses.append(jendl_rmse)

			for o, p in zip(jendlerg, jendlxs):
				if p > 0:
					threshold_values.append(o)
					break

			for x, y in zip(gated_jendl_xs, predjendlgated):
				all_libs.append(x)
				all_preds.append(y)


		if target_nuclide in CENDL_nuclides:
			cendlxs_interpolated = interpolation_function(cendlerg)

			predcendlgated, truncatedcendl, cendl_r2 = r2_standardiser(library_xs=cendlxs,
														   predicted_xs=cendlxs_interpolated)
			cendl_r2s.append(cendl_r2)
			cendl_rmse = mean_squared_error(predcendlgated, truncatedcendl) ** 0.5
			cendl_rmses.append(cendl_rmse)

			for o, p in zip(cendlerg, cendlxs):
				if p > 0:
					threshold_values.append(o)
					break

			for x, y in zip(truncatedcendl, predcendlgated):
				all_libs.append(x)
				all_preds.append(y)


		if target_nuclide in JEFF_nuclides:
			jeffxs_interpolated = interpolation_function(jefferg)

			predjeffgated, d2, jeff_r2 = r2_standardiser(library_xs=jeffxs,
														 predicted_xs=jeffxs_interpolated)
			jeff_r2s.append(jeff_r2)
			jeff_rmse = mean_squared_error(d2, predjeffgated) ** 0.5
			jeff_rmses.append(jeff_rmse)
			for o, p in zip(jefferg, jeffxs):
				if p > 0:
					threshold_values.append(o)
					break
			for x, y in zip(d2, predjeffgated):
				all_libs.append(x)
				all_preds.append(y)



		tendlxs_interpolated = interpolation_function(tendlerg)
		predtendlgated, truncatedtendl, tendlr2 = r2_standardiser(library_xs=tendlxs, predicted_xs=tendlxs_interpolated)
		tendl_r2s.append(tendlr2)
		tendlrmse = mean_squared_error(truncatedtendl, predtendlgated)**0.5
		tendl_rmses.append(tendlrmse)
		for o, p in zip(tendlerg, tendlxs):
			if p > 0:
				threshold_values.append(o)
				break

		for x, y in zip(truncatedtendl, predtendlgated):
			all_libs.append(x)
			all_preds.append(y)


		consensus = r2_score(all_libs, all_preds)

		thr_diffs = []
		for lib_thr in threshold_values:
			difference = lib_thr - predicted_threshold
			thr_diffs.append(difference)
		mean_difference = np.mean(thr_diffs)

		nuclide_thresholds.append(mean_difference)

		consensus_rmse = mean_squared_error(all_libs, all_preds) ** 0.5

		runs_r2_array.append(consensus)

		runs_rmse_array.append(consensus_rmse)


	exp_true_xs = [y for y in y_test]
	exp_pred_xs = [xs for xs in predictions]

	print(f'completed in {time.time() - time1:0.1f} s')

	XS_plot = []
	E_plot = []

	for i, row in enumerate(X_test):
		if [row[0], row[1]] == target_nuclide:
			XS_plot.append(y_test[i])
			E_plot.append(row[4]) # Energy values are in 5th column


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


	print()
	print(f"TENDL-2021: {np.mean(tendl_r2s):0.3f} +- {np.std(tendl_r2s):0.3f}, RMSE {np.mean(tendl_rmses):0.3f} +- {np.std(tendl_rmses):0.3f}")

	junhuaergs = [13.5, 14.1, 14.8]
	junhuaxs = [1.773, 1.772, 1.802]
	junhuadxs = [0.195, 0.218, 0.211]

	filaergs = [13.56, 13.74, 13.96, 14.19, 14.42, 14.61, 14.78]
	filaxs = [1.467, 1.524, 1.526, 1.505, 1.443, 1.544, 1.464]
	filadxs = [0.0680688, 0.0722376, 0.0650076, 0.074046, 0.0656565, 0.0600616, 0.056364]

	jun2erg = [13.5, 14.1, 14.8]
	jun2xs = [1.464, 1.484, 1.533]
	jun2dxs = [0.155, 0.182, 0.177]

	#2sigma CF

	qaime = [14.7]
	qaimxs = [2.047]
	qaimdxs = [0.171]

	konnoe = [13.56, 13.98, 14.22, 14.66, 14.93]
	konnoxs = [1.32, 1.29, 1.28, 1.30, 1.41]
	konnodxs = [0.12, 0.12, 0.12, 0.13, 0.14]

	lue = [14.77]
	luxs = [1.56]
	ludxs = [0.07]

	ikedae = [14.10, 14.50, 14.80]
	ikedaxs = [1.326, 1.533, 1.659]
	ikedadxs = [0.075, 0.077, 0.083]

	wange = [13.5, 13.7, 14.4, 14.6, 14.8]
	wangxs = [1.469, 1.467, 1.502, 1.496, 1.596]
	wangdxs = [0.117, 0.113, 0.118, 0.118, 0.118]

	fila2e = [13.56, 13.74, 13.96, 14.19, 14.42, 14.61, 14.78]
	fila2xs = [1.759, 1.811, 1.809, 1.791, 1.724, 1.819, 1.73]
	fila2dxs = [0.079155, 0.0813139, 0.0750735, 0.0834606, 0.0755112, 0.0713048, 0.067643]

	plt.rcParams.update({'font.size': 12})
	plt.figure()
	plt.rcParams.update({'font.size': 12})
	plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
	plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
	print(f"ENDF/B-VIII: {np.mean(endfb_r2s)} +- {np.std(endfb_r2s)}, RMSE: {np.mean(endfb_rmses):0.3f} +- {np.std(endfb_rmses):0.3f}")
	plt.plot(tendlerg, tendlxs, label = 'TENDL-2021', color='dimgrey')
	if target_nuclide in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
		print(f"JEFF-3.3: {np.mean(jeff_r2s):0.3f} +- {np.std(jeff_r2s):0.3f}")

	if target_nuclide in JENDL_nuclides:
		plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
		print(f"JENDL-5: {np.mean(jendl_r2s):0.3f} +- {np.std(jendl_r2s):0.3f}, RMSE {np.mean(jendl_rmses):0.3f} +- {np.std(jendl_rmses):0.3f}")
	if target_nuclide in CENDL_nuclides:
		plt.plot(cendlerg, cendlxs, '--', label = 'CENDL-3.2', color='gold')
		print(f"CENDL-3.2: {np.mean(cendl_r2s):0.3f} +- {np.std(cendl_r2s):0.3f}, RMSE {np.mean(cendl_rmses):0.3f} +- {np.std(cendl_rmses):0.3f}")
	plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
	# plt.errorbar(junhuaergs, junhuaxs, junhuadxs, fmt='x',
	# 			 capsize=2, label='Junhua, 2017', color='indigo')
	# plt.errorbar(jun2erg, jun2xs, jun2dxs, fmt='x',
	# 			 capsize=2, label='Junhua, 2017', color='violet')
	# plt.errorbar(wange, wangxs, wangdxs, fmt='x',
	# 			 capsize=2, label='Wang, 1992', color='blue')
	# plt.errorbar(konnoe, konnoxs, konnodxs, fmt='x',
	# 			 capsize=2, label='Konno, 1993', color='orangered')
	# plt.errorbar(qaime, qaimxs, qaimdxs, fmt='x',
	# 			 capsize=2, label='Qaim, 1974', color='lightgreen')
	# plt.errorbar(lue, luxs, ludxs, fmt='x',
	# 			 capsize=2, label='Lu, 1991', color='olive')
	# plt.errorbar(filaergs, filaxs, filadxs, fmt='x',
	# 			 capsize=2, label='Filatenkov, 2016', color='aqua')
	# plt.errorbar(fila2e, fila2xs, fila2dxs, fmt='x',
	# 			 capsize=2, label='Filatenkov, 2016', color='slategray')
	plt.grid()
	plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[target_nuclide[0]]}-{target_nuclide[1]}")
	plt.xlabel("Energy / MeV")
	plt.ylabel("$\sigma_{n,2n}$ / b")
	plt.legend(loc='upper left')
	plt.savefig(f'{periodictable.elements[target_nuclide[0]]}-{target_nuclide[1]}_n2n.png', dpi=500)
	plt.show()

	final_runtime = time.time() - runtime

	print(f"Consensus R2: {np.mean(runs_r2_array):0.5f} +/- {np.std(runs_r2_array)}")
	print(f"Consensus RMSE: {np.mean(runs_rmse_array):0.3f} +/- {np.std(runs_rmse_array):0.3f}")

	endfbfunc = scipy.interpolate.interp1d(x=endfberg, y=endfbxs, fill_value='extrapolate')
	e8interptendlgrid = endfbfunc(tendlerg)
	u1, u2, endfb_wrt_tendl = r2_standardiser(tendlxs, e8interptendlgrid)
	print(f'R2 TENDL-2021:ENDF/B-VIII: {endfb_wrt_tendl}:0.3f')
	print(f"Runtime: {timedelta(seconds=final_runtime)}")


print('mean threshold difference:', mean_difference)