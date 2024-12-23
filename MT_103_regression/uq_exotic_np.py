import time
import pandas as pd
import tqdm
import xgboost as xg
from matrix_functions import range_setter, r2_standardiser
import random
from MT_103_functions import make_test, make_train, Generalplotter103
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import periodictable
import scipy
import numpy as np


df = pd.read_csv('ENDFBVIII_MT_103_all_features.csv')
df = df[(df['Z'] != 6) & (df['A'] != 12)]
df.index = range(len(df))
CENDL_32 = pd.read_csv('CENDL-3.2_MT_103_all_features.csv')
CENDL_nuclides = range_setter(df=CENDL_32, la=0, ua=208)
#
JEFF_33 = pd.read_csv('JEFF-3.3_MT_103_all_features.csv')
JEFF_nuclides = range_setter(df=JEFF_33, la=0, ua=208)
#
JENDL_5 = pd.read_csv('JENDL-5_MT_103_all_features.csv')
JENDL_nuclides = range_setter(df=JENDL_5, la=0, ua=208)

#
TENDL_2021 = pd.read_csv('TENDL-2021_MT_103_all_features.csv')
TENDL_nuclides = range_setter(df=TENDL_2021, la=0, ua=208)

ENDFB_nuclides = range_setter(df=df, la=0, ua=208)
print("Data loaded...")


validation_set_size = 1

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


target_nuclide = [63,158]


runs_r2_array = []




n_evaluations = 5
datapoint_matrix = []

validation_nuclides = [target_nuclide]
X_test, y_test = make_test(validation_nuclides, df=TENDL_2021, minerg=minenergy, maxerg=maxenergy)

X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides,
									la=0, ua=208, maxerg=maxenergy, minerg=minenergy,
							  mode='both')

jendl_r2s = []
tendl_r2s = []

for i in tqdm.tqdm(range(n_evaluations)):




	model_seed = random.randint(a=1, b=1000)



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

		temp_x, temp_y = make_test(nuclides=[n], df=TENDL_2021, minerg=minenergy, maxerg=maxenergy)
		initial_predictions = model.predict(temp_x)

		for p in initial_predictions:
			if p >= (0.0 * max(initial_predictions)):
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

		cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL_32)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(predicted_xs=pred_cendl,
																		   library_xs=cendl_xs)

		pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
		for x, y in zip(pred_cendl_gated, truncated_cendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if target_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL_5, minerg=minenergy, maxerg=maxenergy)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(predicted_xs=pred_jendl,
																		   library_xs=jendl_xs)

		jendl_r2s.append(pred_jendl_r2)

		for x, y in zip(pred_jendl_gated, truncated_jendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF_33, minerg=minenergy, maxerg=maxenergy)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(predicted_xs=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if target_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL_2021, minerg=minenergy, maxerg=maxenergy)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(predicted_xs=pred_tendl,
																		   library_xs=tendl_xs)
		tendl_r2s.append(pred_tendl_r2)
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


tendl_energy, tendl_xs = Generalplotter103(dataframe=TENDL_2021, nuclide=target_nuclide)
jendlerg, jendlxs = Generalplotter103(dataframe=JENDL_5, nuclide=target_nuclide, maxenergy=20)
cendlerg, cendlxs = Generalplotter103(dataframe=CENDL_32, nuclide=target_nuclide)
jefferg, jeffxs = Generalplotter103(dataframe=JEFF_33, nuclide=target_nuclide)
tendlerg, tendlxs = Generalplotter103(dataframe=TENDL_2021, nuclide=target_nuclide)

# 2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
if target_nuclide in ENDFB_nuclides:
	plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendl_energy, tendl_xs, label = 'TENDL-2021', color='dimgrey', linewidth=2)
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(jendlerg, jendlxs, '--', label='JENDL-5', color='green')
if target_nuclide in CENDL_nuclides:
	plt.plot(cendlerg, cendlxs, '--', label = 'CENDL-3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
plt.grid()
plt.title(fr"$\sigma_{{n,p}}$ for {periodictable.elements[target_nuclide[0]]}-{target_nuclide[1]}")
plt.xlabel("Energy / MeV")
plt.ylabel(r"$\sigma_{n,p}$ / b")
plt.legend(loc='upper left')
plt.show()

mu_r2 = np.mean(runs_r2_array)
sigma_r2 = np.std(runs_r2_array)
interval_low_r2, interval_high_r2 = scipy.stats.t.interval(0.95, loc=mu_r2, scale=sigma_r2, df=(len(runs_r2_array) - 1))
print(f"\nConsensus: {mu_r2:0.5f} +/- {interval_high_r2-interval_low_r2:0.5f}")
print(f'Mass difference: {diff(target_nuclide)}')

print(np.mean(jendl_r2s), np.std(jendl_r2s), 'jendl')
print(np.mean(tendl_r2s), np.std(tendl_r2s), 'tendl')