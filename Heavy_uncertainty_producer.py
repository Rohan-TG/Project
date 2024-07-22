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
	range_setter, General_plotter, r2_standardiser
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")



al = range_setter(la=210, ua=260, df=df)




n_evaluations = 10
datapoint_matrix = []
target_nuclide = [92,235]

for i in tqdm.tqdm(range(n_evaluations)):
	# print(f"\nRun {i+1}/{n_evaluations}")


	validation_nuclides = [target_nuclide]
	validation_set_size = 5  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	# print("\nTest nuclide selection complete")

	time1 = time.time()

	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=210, ua=260,) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=df,)

	# print("Data prep done")

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=500,
							learning_rate=0.01,
							max_depth=8,
							subsample=0.2,
							max_leaves=0,
							seed=model_seed,)


	model.fit(X_train, y_train)

	# print("Training complete")

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

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i] # validation nuclide
		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide

	exp_true_xs = [y for y in y_test]
	exp_pred_xs = [xs for xs in predictions]

	# print(f"MSE: {mean_squared_error(exp_true_xs, exp_pred_xs, squared=False)}") # MSE
	# print(f"R2: {r2_score(exp_true_xs, exp_pred_xs):0.5f}") # Total R^2 for all predictions in this training campaign
	# print(f'completed in {time.time() - time1:0.1f} s')


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
TENDL_nuclides = range_setter(df=TENDL, la=210, ua=260)
tendl_energy, tendl_xs = General_plotter(df=TENDL, nuclides=[target_nuclide])


JEFF33 = pd.read_csv('JEFF33_all_features.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=210, ua=260)
JEFF_energy, JEFF_XS = General_plotter(df=JEFF33, nuclides=[target_nuclide])


JENDL5 = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL5_energy, JENDL5_XS = General_plotter(df=JENDL5, nuclides=[target_nuclide])
JENDL_nuclides = range_setter(df=JENDL5, la=210, ua=260)


CENDL32 = pd.read_csv('CENDL32_all_features.csv')
CENDL32.index = range(len(CENDL32))
CENDL32_energy, CENDL33_XS = General_plotter(df=CENDL32, nuclides=[target_nuclide])
CENDL_nuclides = range_setter(df=CENDL32, la=210, ua=260)
mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)

pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
d1, d2, pred_endfb_r2 = r2_standardiser(predicted_xs=pred_xs, library_xs=true_xs)
print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

if target_nuclide in CENDL_nuclides:
	cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL32)
	pred_cendl = model.predict(cendl_test)

	pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
	d1, d2, pred_cendl_r2 = r2_standardiser(predicted_xs=pred_cendl, library_xs=cendl_xs)
	print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

if target_nuclide in JENDL_nuclides:
	jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL5)
	pred_jendl = model.predict(jendl_test)

	pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
	d1, d2, pred_jendl_r2 = r2_standardiser(predicted_xs=pred_jendl, library_xs=jendl_xs)
	print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")

if target_nuclide in JEFF_nuclides:
	jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF33)

	pred_jeff = model.predict(jeff_test)

	d1, d2, pred_jeff_r2 = r2_standardiser(predicted_xs=pred_jeff, library_xs=jeff_xs)
	print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f}")

if target_nuclide in TENDL_nuclides:
	tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL)
	pred_tendl = model.predict(tendl_test)

	d1, d2, pred_tendl_r2 = r2_standardiser(predicted_xs=pred_tendl, library_xs=tendl_xs)
	print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f}")



print(f"Turning points: {dsigma_dE(XS=datapoint_means)}")

#2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendl_energy, tendl_xs, label = 'TENDL21', color='dimgrey')
if target_nuclide in JEFF_nuclides:
	plt.plot(JEFF_energy, JEFF_XS, '--', label='JEFF3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(JENDL5_energy, JENDL5_XS, label='JENDL5', color='green')
if target_nuclide in CENDL_nuclides:
	plt.plot(CENDL32_energy, CENDL33_XS, '--', label = 'CENDL3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')

plt.grid()
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.legend(loc='upper left')
plt.show()


final_runtime = time.time() - runtime

print(f"Runtime: {timedelta(seconds=final_runtime)}")
