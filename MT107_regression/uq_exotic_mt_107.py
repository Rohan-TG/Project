import time
import pandas as pd
import xgboost as xg
from matrix_functions import range_setter, r2_standardiser
import random
from functions107 import maketest107, maketrain107, Generalplotter107
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import periodictable
import scipy
import numpy as np


df = pd.read_csv('ENDFBVIII_MT_107_all_features.csv')
df = df[(df['Z'] != 6) & (df['A'] != 12)]
df.index = range(len(df))
CENDL_32 = pd.read_csv('CENDL-3.2_MT_107_all_features.csv')
CENDL_nuclides = range_setter(df=CENDL_32, la=0, ua=208)
#
JEFF_33 = pd.read_csv('JEFF-3.3_MT_107_all_features.csv')
JEFF_nuclides = range_setter(df=JEFF_33, la=0, ua=208)
#
JENDL_5 = pd.read_csv('JENDL-5_MT_107_all_features.csv')
JENDL_nuclides = range_setter(df=JENDL_5, la=0, ua=208)

#
TENDL_2021 = pd.read_csv('TENDL-2021_MT_107_all_features.csv')
TENDL_nuclides = range_setter(df=TENDL_2021, la=0, ua=208)

ENDFB_nuclides = range_setter(df=df, la=0, ua=208)
print("Data loaded...")


validation_set_size = 10

minenergy = 0.1
maxenergy = 20

def diff(target):
	match_element = []
	zl = [n[0] for n in ENDFB_nuclides]
	if target[0] in zl:
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


target_nuclide = []

exotic_nuclides = [[target_nuclide]]
runs_r2_array = []
for n in TENDL_nuclides:
	if n not in ENDFB_nuclides:
		exotic_nuclides.append(n)




n_evaluations = 10
datapoint_matrix = []

for i in range(n_evaluations):

	validation_nuclides = []
	while len(validation_nuclides) < validation_set_size:
		nuclide_choice = random.choice(
			exotic_nuclides)  # randomly select nuclide from list of all nuclides in ENDF/B-VIII
		if nuclide_choice not in validation_nuclides:
			validation_nuclides.append(nuclide_choice)


	model_seed = random.randint(a=1, b=1000)
	X_train, y_train = maketrain107(df=df, validation_nuclides=validation_nuclides,
									la=0, ua=208, maxerg=maxenergy, minerg=minenergy)
	X_test, y_test = maketest107(validation_nuclides, df=TENDL_2021)

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.01,
							max_depth=7,
							subsample=0.2,
							reg_lambda=2,
							seed=model_seed
							)

	model.fit(X_train, y_train)

	predictions_r = []
	for n in validation_nuclides:

		temp_x, temp_y = maketest107(nuclides=[n], df=TENDL_2021)
		initial_predictions = model.predict(temp_x)

		for p in initial_predictions:
			if p >= (0.02 * max(initial_predictions)):
				predictions_r.append(p)
			else:
				predictions_r.append(0.0)

	predictions = predictions_r

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

	all_preds = []
	all_libs = []

	if target_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = maketest107(nuclides=[target_nuclide], df=CENDL_32)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(predicted_xs=pred_cendl,
																		   library_xs=cendl_xs)

		pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
		for x, y in zip(pred_cendl_gated, truncated_cendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if target_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = maketest107(nuclides=[target_nuclide], df=JENDL_5)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(predicted_xs=pred_jendl,
																		   library_xs=jendl_xs)

		for x, y in zip(pred_jendl_gated, truncated_jendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = maketest107(nuclides=[target_nuclide], df=JEFF_33)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(predicted_xs=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if target_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = maketest107(nuclides=[target_nuclide], df=TENDL_2021)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(predicted_xs=pred_tendl,
																		   library_xs=tendl_xs)
		for x, y in zip(pred_tendl_gated, truncated_tendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

	consensus = r2_score(all_libs, all_preds)



	runs_r2_array.append(consensus)


XS_plot = []
E_plot = []

for i, row in enumerate(X_test):
	if [row[0], row[1]] == validation_nuclides[0]:
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
	datapoint_means.append(mu)
	datapoint_upper_interval.append(interval_high)
	datapoint_lower_interval.append(interval_low)



lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point-low)
	upper_bound.append(point+up)


tendl_energy, tendl_xs = Generalplotter107(dataframe=TENDL_2021, nuclide=target_nuclide)

