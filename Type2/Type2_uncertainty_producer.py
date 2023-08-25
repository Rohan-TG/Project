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
	range_setter, General_plotter
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

CENDL32 = pd.read_csv('CENDL32_all_features.csv')
CENDL32.index = range(len(CENDL32))

JENDL5 = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL5.index = range(len(JENDL5))

JEFF33 = pd.read_csv('JEFF33_all_features.csv')
JEFF33.index = range(len(JEFF33))

TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]





all_libraries = pd.concat([df, JENDL5])

# all_libraries = pd.concat([all_libraries, tendl21])
all_libraries = pd.concat([all_libraries, JEFF33])
all_libraries = pd.concat([all_libraries, CENDL32])

nucs = range_setter(df=df, la=30, ua=210)

all_libraries.index = range(len(all_libraries))

df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))


al = range_setter(la=30, ua=210, df=df)




n_evaluations = 50
o_nucs = [[40,85], [40,86], [40,87],
		  [40,88], [40,89], [40,95],
		  [40,97], [40,98], [40,99],
		  [40,100], [40,101],[40,102]]


for x in o_nucs:
	target_nuclide = x
	datapoint_matrix = []

	for i in tqdm.tqdm(range(n_evaluations)):
		print(f"\nRun {i+1}/{n_evaluations}")


		validation_nuclides = [target_nuclide]
		validation_set_size = 5  # number of nuclides hidden from training

		while len(validation_nuclides) < validation_set_size:
			choice = random.choice(nucs)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides:
				validation_nuclides.append(choice)
		print("\nTest nuclide selection complete")

		time1 = time.time()

		X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides, la=30, ua=210,) # make training matrix

		X_test, y_test = make_test(validation_nuclides, df=TENDL,)

		print("Data prep done")

		model_seed = random.randint(a=1, b=1000) # seed for subsampling

		model = xg.XGBRegressor(n_estimators=900,
								reg_lambda=0.9530201929544536,
								learning_rate=0.005,
								max_depth=12,
								subsample=0.15315395901090592,
								seed=model_seed)


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




	TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210,)
	tendl_energy, tendl_xs = General_plotter(df=TENDL, nuclides=[target_nuclide])



	JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=210,)
	JEFF_energy, JEFF_XS = General_plotter(df=JEFF33, nuclides=[target_nuclide])



	JENDL5_energy, JENDL5_XS = General_plotter(df=JENDL5, nuclides=[target_nuclide])
	JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=210,)



	CENDL32_energy, CENDL33_XS = General_plotter(df=CENDL32, nuclides=[target_nuclide])
	CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=210,)
	mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)
	r2 = r2_score(true_xs, pred_xs)  # R^2 score for this specific nuclide
	print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}, MSE: {mse:0.5f}")
	time.sleep(0.8)

	pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
	pred_endfb_r2 = r2_score(y_true=true_xs, y_pred=pred_xs)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

	if target_nuclide in CENDL_nuclides:
		cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL32)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
		pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=pred_cendl)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if target_nuclide in JENDL_nuclides:
		jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL5)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=pred_jendl)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF33)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_r2 = r2_score(y_true=jeff_xs, y_pred=pred_jeff)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f}")

	if target_nuclide in TENDL_nuclides:
		tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=pred_tendl)
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
