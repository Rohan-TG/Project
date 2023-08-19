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
from matrix_functions import anomaly_remover, log_make_train, log_make_test,\
	range_setter, General_plotter
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]

df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))


df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=30, ua=215, df=df)


log_reduction_var = 0.00001

n_evaluations = 100
datapoint_matrix = []
target_nuclide = [20,40]

for i in tqdm.tqdm(range(n_evaluations)):
	print(f"\nRun {i+1}/{n_evaluations}")


	validation_nuclides = [target_nuclide]
	validation_set_size = 20  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("\nTest nuclide selection complete")

	time1 = time.time()

	X_train, y_train = log_make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215,
									  log_reduction_variable=log_reduction_var) # make training matrix

	X_test, y_test = log_make_test(validation_nuclides, df=df_test,
								   log_reduction_variable=log_reduction_var)

	print("Data prep done")

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.01,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=model_seed,)


	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions
	new_pred = []
	for p in predictions:
		new_pred.append(np.exp(p) - log_reduction_var)
	predictions = new_pred

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
				dummy_test_XS.append(np.exp(y_test[i]) - log_reduction_var)
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i] # validation nuclide
		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide

	exp_true_xs = [y for y in y_test]
	exp_pred_xs = [xs for xs in predictions]

	print(f"MSE: {mean_squared_error(exp_true_xs, exp_pred_xs, squared=False)}") # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs):0.5f}") # Total R^2 for all predictions in this training campaign
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



TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)
tendl_energy, tendl_xs = General_plotter(df=TENDL, nuclides=[target_nuclide])
time.sleep(2)


JEFF33 = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=210)
JEFF_energy, JEFF_XS = General_plotter(df=JEFF33, nuclides=[target_nuclide])
time.sleep(2)

JENDL5 = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL5_energy, JENDL5_XS = General_plotter(df=JENDL5, nuclides=[target_nuclide])
JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=210)
time.sleep(2)

CENDL32 = pd.read_csv('CENDL32_features_arange_zeroed.csv')
CENDL32.index = range(len(CENDL32))
CENDL32_energy, CENDL33_XS = General_plotter(df=CENDL32, nuclides=[target_nuclide])
CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=210)
time.sleep(2)

f_predictions = scipy.interpolate.interp1d(x=E_plot, y=datapoint_means, fill_value='extrapolate')

if target_nuclide in CENDL_nuclides:
	cendl_erg, cendl_xs = General_plotter(df=CENDL32, nuclides=[target_nuclide])
	x_interpolate_cendl = np.linspace(start=0.0, stop=max(cendl_erg), num=500)
	predictions_interpolated_cendl = f_predictions(x_interpolate_cendl)

	f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs)
	cendl_interp_xs = f_cendl32(x_interpolate_cendl)
	pred_cendl_mse = mean_squared_error(predictions_interpolated_cendl, cendl_interp_xs)
	pred_cendl_r2 = r2_score(y_true=cendl_interp_xs, y_pred=predictions_interpolated_cendl)
	print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

if target_nuclide in JENDL_nuclides:
	jendl_erg, jendl_xs = General_plotter(df=JENDL5, nuclides=[target_nuclide])
	x_interpolate_jendl = np.linspace(start=0.0, stop=max(jendl_erg), num=500)
	predictions_interpolated_jendl = f_predictions(x_interpolate_jendl)

	f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs)
	jendl_interp_xs = f_jendl5(x_interpolate_jendl)
	pred_jendl_mse = mean_squared_error(predictions_interpolated_jendl, jendl_interp_xs)
	pred_jendl_r2 = r2_score(y_true=jendl_interp_xs, y_pred=predictions_interpolated_jendl)
	print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

if target_nuclide in JEFF_nuclides:
	jeff_erg, jeff_xs = General_plotter(df=JEFF33, nuclides=[target_nuclide])
	x_interpolate_jeff = np.linspace(start=0.0, stop=max(jeff_erg), num=500)
	predictions_interpolated_jeff = f_predictions(x_interpolate_jeff)

	f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs)
	jeff_interp_xs = f_jeff33(x_interpolate_jeff)
	pred_jeff_mse = mean_squared_error(predictions_interpolated_jeff, jeff_interp_xs)
	pred_jeff_r2 = r2_score(y_true=jeff_interp_xs, y_pred=predictions_interpolated_jeff)
	print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

if target_nuclide in TENDL_nuclides:
	tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[target_nuclide])
	x_interpolate_tendl = np.linspace(start=0.0, stop=max(tendl_erg), num=500)
	predictions_interpolated_tendl = f_predictions(x_interpolate_tendl)

	f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs)
	tendl_interp_xs = f_tendl21(x_interpolate_tendl)
	pred_tendl_mse = mean_squared_error(predictions_interpolated_tendl, tendl_interp_xs)
	pred_tendl_r2 = r2_score(y_true=tendl_interp_xs, y_pred=predictions_interpolated_tendl)
	print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")



title_string_latex = "$\sigma_{n,2n}$"
title_string_nuclide = f"for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}"
title_string = title_string_latex+title_string_nuclide

endfb_erg, endfb_xs = General_plotter(df=df, nuclides=[validation_nuclides[0]])

#2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(endfb_erg, endfb_xs, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendl_energy, tendl_xs, label = 'TENDL21', color='dimgrey')
plt.plot(JEFF_energy, JEFF_XS, label='JEFF3.3', color='mediumvioletred')
plt.plot(JENDL5_energy, JENDL5_XS, '--', label='JENDL5', color='green')
plt.plot(CENDL32_energy, CENDL33_XS, '--', label = 'CENDL3.3', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
plt.grid()
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.legend(loc='upper left')
plt.show()
time.sleep(1)


final_runtime = time.time() - runtime

print(f"Runtime: {timedelta(seconds=final_runtime)}")
