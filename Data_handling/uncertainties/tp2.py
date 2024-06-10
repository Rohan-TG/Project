import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matrix_functions import range_setter, r2_standardiser, General_plotter, exclusion_func
from propagator_functions import make_train_sampler, make_test
from sklearn.metrics import r2_score, mean_squared_error
import time
import periodictable
from datetime import timedelta
import tqdm
import random
runtime = time.time()

ENDFBVIII = pd.read_csv('ENDFBVIII_100keV_all_uncertainties.csv')
ENDFB_nuclides = range_setter(df=ENDFBVIII, la=0, ua= 210)

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
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=210)




exc = exclusion_func() # 10 sigma with handpicked additions



target_nuclide = [82,208]
n_evaluations = 2

jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])
endfberg, endfbxs = General_plotter(df=ENDFBVIII, nuclides=[target_nuclide])


validation_set_size = 20




runs_r2_array = []

datapoint_matrix = []

print('Data loaded...')

endfb_r2s = []
cendl_r2s = []
tendl_r2s = []
jeff_r2s = []
jendl_r2s = []

for i in tqdm.tqdm(range(n_evaluations)):
	# print(f"\nRun {i + 1}/{n_evaluations}")
	model_seed = random.randint(a=1, b=10000)
	target_nuclides = [target_nuclide]

	while len(target_nuclides) < validation_set_size:  # up to 25 nuclides
		choice = random.choice(ENDFB_nuclides)  # randomly select nuclide from list of all nuclides in ENDF/B-VIII
		if choice not in target_nuclides:
			target_nuclides.append(choice)
	print("Test nuclide selection complete")


	time1 = time.time()
	X_test, y_test = make_test(nuclides=target_nuclides, df=ENDFBVIII)
	print('Test data generation complete...')

	X_train, y_train = make_train_sampler(df=ENDFBVIII, la=30, ua=210,
										validation_nuclides=target_nuclides, exclusions=exc)
	print("Training...")

	model = xg.XGBRegressor(n_estimators=950,  # define regressor
							learning_rate=0.008,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=model_seed,)

	model.fit(X_train, y_train)
	print("Training complete")

	predictions_ReLU = []

	for n in target_nuclides:
		tempx, tempy = make_test(nuclides=[n], df=ENDFBVIII)
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

	endfbgated, dee, pred_endfb_r2 = r2_standardiser(library_xs=XS_plotmatrix[0], predicted_xs=P_plotmatrix[0])
	endfb_r2s.append(r2_score(endfbgated, dee))
	for x, y in zip(dee, endfbgated):
		all_libs.append(x)
		all_preds.append(y)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

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
print(f"TENDL-2021: {np.mean(tendl_r2s)} +- {np.std(tendl_r2s)}")




Bayhurst_energies = [8.51000E+00, 9.27000E+00, 1.41000E+01, 1.49200E+01, 1.71900E+01, 1.81900E+01, 1.99400E+01, 2.12100E+01, 2.19400E+01, 2.33200E+01, 2.44600E+01]
Bayhurst_XS = [3.23000E-01, 8.16000E-01, 1.78900E+00, 1.66800E+00, 1.33700E+00, 9.65000E-01, 5.74000E-01, 3.03000E-01, 3.42000E-01, 2.87000E-01, 2.46000E-01]
Bayhurst_delta_XS = [1.50000E-02, 3.50000E-02, 7.60000E-02, 7.10000E-02, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.42410E-02, 1.57320E-02, 1.42065E-02, 1.32102E-02]
# Reference:





Veeser_energies = [1.47000E+01, 1.60000E+01, 1.70000E+01, 1.80000E+01, 1.90000E+01, 2.00000E+01, 2.10000E+01, 2.20000E+01, 2.30000E+01, 2.40000E+01]
Veeser_XS = [1.98400E+00, 1.99200E+00, 1.78700E+00, 1.32000E+00, 9.57000E-01, 7.70000E-01, 7.22000E-01, 5.35000E-01, 5.62000E-01, 5.24000E-01]
Veeser_delta_XS = [1.15000E-01, 1.05000E-01, 8.50000E-02, 8.80000E-02, 6.40000E-02, 1.04000E-01, 6.80000E-02, 7.90000E-02, 9.80000E-02, 1.54000E-01]


Dzysiuk_energies =[1.35000E+01, 1.42000E+01, 1.46000E+01]
Dzysiuk_XS =[1.89600E+00, 1.47300E+00, 1.86000E+00]
Dzysiuk_delta_XS = [2.50000E-01, 2.19000E-01, 1.90000E-01]


#
# Frehaut_E = [8.44000E+00, 8.94000E+00, 9.44000E+00, 9.93000E+00, 1.04200E+01, 1.09100E+01, 1.14000E+01, 1.18800E+01, 1.23600E+01, 1.28500E+01, 1.33300E+01, 1.38000E+01, 1.42800E+01, 1.47600E+01]
# Frehaut_XS = [2.03000E-01, 6.59000E-01, 1.16200E+00, 1.40400E+00, 1.56000E+00, 1.72000E+00, 1.69600E+00, 1.81300E+00, 1.91900E+00, 1.99600E+00, 2.03600E+00, 2.07600E+00, 2.11300E+00, 2.09400E+00]
# Frehaut_XS_d = [3.30000E-02, 5.00000E-02, 1.12000E-01, 8.60000E-02, 9.50000E-02, 1.24000E-01, 1.07000E-01, 1.10000E-01, 1.21000E-01, 1.25000E-01, 1.26000E-01, 1.23000E-01, 1.55000E-01, 1.57000E-01]

Frehaut_E = [7.93000E+00, 8.18000E+00, 8.44000E+00, 8.94000E+00, 9.44000E+00, 9.93000E+00, 1.04200E+01, 1.09100E+01, 1.14000E+01, 1.18800E+01, 1.23600E+01, 1.28500E+01, 1.33300E+01, 1.38000E+01, 1.42800E+01, 1.47600E+01]
Frehaut_XS = [8.30000E-02, 1.71000E-01, 3.14000E-01, 6.05000E-01, 9.41000E-01, 1.18400E+00, 1.43400E+00, 1.58000E+00, 1.71400E+00, 1.80400E+00, 1.81700E+00, 1.92100E+00, 1.97200E+00, 2.07100E+00, 2.01000E+00, 2.02800E+00]
Frehaut_XS_d = [1.00000E-02, 1.40000E-02, 2.20000E-02, 4.00000E-02, 6.10000E-02, 7.50000E-02, 8.30000E-02, 9.70000E-02, 9.80000E-02, 1.05000E-01, 1.37000E-01, 1.43000E-01, 1.50000E-01, 1.60000E-01, 1.58000E-01, 1.63000E-01]

#2sigma CF
plt.figure()
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
print(f"ENDF/B-VIII: {np.mean(endfb_r2s)} +- {np.std(endfb_r2s)}")
plt.plot(tendlerg, tendlxs, label = 'TENDL-2021', color='dimgrey')
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
	print(f"JEFF-3.3: {np.mean(jeff_r2s)} +- {np.std(jeff_r2s)}")
if target_nuclide in JENDL_nuclides:
	plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
	print(f"JENDL-5: {np.mean(jendl_r2s)} +- {np.std(jendl_r2s)}")
if target_nuclide in CENDL_nuclides:
	plt.plot(cendlerg, cendlxs, '--', label = 'CENDL-3.2', color='gold')
	print(f"CENDL-3.2: {np.mean(cendl_r2s)} +- {np.std(cendl_r2s)}")
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
# plt.errorbar(Bayhurst_energies, Bayhurst_XS, Bayhurst_delta_XS, fmt='x',
# 			 capsize=2, label='Bayhurst, 1975', color='indigo')
plt.errorbar(Frehaut_E, Frehaut_XS, Frehaut_XS_d, fmt='x',
			 capsize=2, label='Frehaut, 1980', color='violet')
# plt.errorbar(Dzysiuk_energies, Dzysiuk_XS, Dzysiuk_delta_XS, fmt='x',
# 			 capsize=2, label='Dzysiuk, 2010', color='blue')
# plt.errorbar(Veeser_energies, Veeser_XS, Veeser_delta_XS, fmt='x',
# 			 capsize=2, label='Veeser, 1977', color='orangered')
plt.grid()
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[target_nuclide[0]]}-{target_nuclide[1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.legend(loc='upper left')
plt.show()

final_runtime = time.time() - runtime

print(f"Consensus R2: {np.mean(runs_r2_array):0.5f} +/- {np.std(runs_r2_array)}")

print()
print(f"Runtime: {timedelta(seconds=final_runtime)}")