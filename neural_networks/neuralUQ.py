from matrix_functions import r2_standardiser
import pandas as pd
import keras
import random
import tensorflow as tf
import time
from datetime import timedelta
from neural_matrix_functions import make_train, make_test, range_setter, General_plotter
import periodictable
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import tqdm
from sklearn.metrics import mean_squared_error, r2_score
runtime = time.time()
JENDL = pd.read_csv('JENDL5_with_MT16_and_ENDFB_MT103_107.csv')
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=208)

TENDL = pd.read_csv('TENDL_2021_MT16_with_ENDFB_MT103_107.csv')
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=208)

CENDL = pd.read_csv('CENDL32_MT16_with_ENDFB_MT103_107.csv')
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=208)

JEFF = pd.read_csv('JEFF33_all_features_MT16_103_107.csv')
JEFF_nuclides = range_setter(df=JEFF, la=30, ua = 208)

df = pd.read_csv('ENDFBVIII_MT16_91_103_107.csv')
ENDFB_nuclides = range_setter(df=df, la=30, ua=208)


print('Data loaded')






callback = keras.callbacks.EarlyStopping(monitor='loss',
										 min_delta=0.005,
										 patience=20,
										 mode='min',
										 start_from_epoch=20)



n_evals = 10
datapoint_matrix = []

target_nuclide = [40,90]

jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])
endfberg, endfbxs = General_plotter(df=df, nuclides=[target_nuclide])


runs_r2_array = []
runs_rmse_array = []

endfb_r2s = []
endfb_rmses = []

jendl_r2s = []
jendl_rmses = []

cendl_r2s = []
cendl_rmses = []

tendl_r2s=[]
tendl_rmses  = []

jeff_r2s=[]
jeff_rmses = []


for i in tqdm.tqdm(range(n_evals)):

	validation_nuclides = [target_nuclide]
	validation_set_size = 20

	while len(validation_nuclides) < validation_set_size:  # up to 25 nuclides
		choice = random.choice(ENDFB_nuclides)  # randomly select nuclide from list of all nuclides in ENDF/B-VIII
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)

	X_train, y_train = make_train(df=df, validation_nuclides= validation_nuclides, la=30, ua=208)

	X_test, y_test = make_test(nuclides=validation_nuclides, df=df)
	model = keras.Sequential()
	model.add(keras.layers.Dense(100, input_shape=(17,), kernel_initializer='normal', activation='relu',
								 ))
	model.add(keras.layers.Dense(400, activation='relu'))
	model.add(keras.layers.Dense(800, activation='relu'))
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Dense(1022, activation='relu'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(400, activation='relu'))
	model.add(keras.layers.Dense(1, activation='linear',))

	model.compile(loss='mean_squared_error', optimizer='adam')

	history = model.fit(
		X_train,
		y_train,
		epochs=100,
		batch_size= 100,
		callbacks=callback,
		validation_split = 0.2,
		verbose=1,)

	predictions_ReLU = []

	for n in validation_nuclides:

		temp_x, temp_y = make_test(nuclides=[n], df=df)
		initial_predictions = model.predict(temp_x)
		initial_predictions = initial_predictions.ravel()

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
				dummy_test_E.append(row[2])  # Energy values are in 5th row
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
	# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

	interpolation_function = scipy.interpolate.interp1d(endfberg, y=P_plotmatrix[0],
														fill_value='extrapolate')

	if target_nuclide in JENDL_nuclides:
		jendlxs_interpolated = interpolation_function(jendlerg)

		predjendlgated, gated_jendl_xs, jendl_r2 = r2_standardiser(library_xs=jendlxs,
																   predicted_xs=jendlxs_interpolated)
		jendl_r2s.append(jendl_r2)
		jendl_rmse = mean_squared_error(predjendlgated, gated_jendl_xs) ** 0.5
		jendl_rmses.append(jendl_rmse)

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
		for x, y in zip(d2, predjeffgated):
			all_libs.append(x)
			all_preds.append(y)

	tendlxs_interpolated = interpolation_function(tendlerg)
	predtendlgated, truncatedtendl, tendlr2 = r2_standardiser(library_xs=tendlxs, predicted_xs=tendlxs_interpolated)
	tendl_r2s.append(tendlr2)
	tendlrmse = mean_squared_error(truncatedtendl, predtendlgated) ** 0.5
	tendl_rmses.append(tendlrmse)
	for x, y in zip(truncatedtendl, predtendlgated):
		all_libs.append(x)
		all_preds.append(y)

	consensus = r2_score(all_libs, all_preds)

	consensus_rmse = mean_squared_error(all_libs, all_preds) ** 0.5

	runs_r2_array.append(consensus)

	runs_rmse_array.append(consensus_rmse)

exp_true_xs = [y for y in y_test]
exp_pred_xs = [xs for xs in predictions]


XS_plot = []
E_plot = []

for i, row in enumerate(X_test):
	if [row[0], row[1]] == target_nuclide:
		XS_plot.append(y_test[i])
		E_plot.append(row[4])  # Energy values are in 5th column

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

lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point - low)
	upper_bound.append(point + up)

plt.figure()
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
plt.show()

final_runtime = time.time() - runtime

print(f"Consensus R2: {np.mean(runs_r2_array):0.5f} +/- {np.std(runs_r2_array)}")
print(f"Consensus RMSE: {np.mean(runs_rmse_array):0.3f} +/- {np.std(runs_rmse_array):0.3f}")

endfbfunc = scipy.interpolate.interp1d(x=endfberg, y=endfbxs, fill_value='extrapolate')
e8interptendlgrid = endfbfunc(tendlerg)
u1, u2, endfb_wrt_tendl = r2_standardiser(tendlxs, e8interptendlgrid)
print(f'R2 TENDL-2021:ENDF/B-VIII: {endfb_wrt_tendl}:0.3f')
print(f"Runtime: {timedelta(seconds=final_runtime)}")