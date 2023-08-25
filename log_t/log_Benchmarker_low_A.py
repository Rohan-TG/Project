import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
import scipy
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import log_make_test, log_make_train, range_setter, General_plotter

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")  # dataframe as above, but with the new features from the Gilbert-Cameron model


TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=60)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=60)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=60)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=60)


al = range_setter(la=0, ua=60, df=df)

log_reduction_var = 0.00001

nuclides_used = []

nuclide_mse = []
nuclide_r2 = []
all_library_evaluations = []
all_interpolated_predictions = []

every_prediction_list = []
every_true_value_list = []

counter = 0
while len(nuclides_used) < len(al):

	validation_nuclides = []  # list of nuclides used for validation
	# test_nuclides = []
	validation_set_size = 10  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:

		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides and choice not in nuclides_used:
			validation_nuclides.append(choice)
			nuclides_used.append(choice)
		if len(nuclides_used) == len(al):
			break

	print("Test nuclide selection complete")
	print(f"{len(nuclides_used)}/{len(al)} selections")


	X_train, y_train = log_make_train(df=df, validation_nuclides=validation_nuclides,
									  la=0, ua=60, log_reduction_variable=log_reduction_var) # make training matrix

	X_test, y_test = log_make_test(validation_nuclides, df=df_test,
								   log_reduction_variable=log_reduction_var)

	print("Train/val matrices generated")


	model = xg.XGBRegressor(n_estimators=500,
							learning_rate=0.007,
							max_depth=8,
							subsample=0.2,
							max_leaves=0,
							seed=42,)

	time1 = time.time()
	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions

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
				dummy_predictions.append(np.exp(predictions[i]) - log_reduction_var)

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	batch_r2 = []
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i]

		nuc = validation_nuclides[i] # validation nuclide

		evaluation_r2s = []

		# endf_erg, endf_xs = General_plotter(df=df, nuclides=[nuc])

		for libxs, p in zip(true_xs, pred_xs):
			all_library_evaluations.append(libxs)
			all_interpolated_predictions.append(p)

		if nuc in CENDL_nuclides:
			cendl_test, cendl_xs = log_make_test(nuclides=[nuc], df=CENDL,
												 log_reduction_variable=log_reduction_var)

			log_preds_cendl = model.predict(cendl_test)
			cendl_xs = [(np.exp(k) - log_reduction_var) for k in cendl_xs]
			preds_cendl = [(np.exp(k) - log_reduction_var) for k in log_preds_cendl]

			pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=preds_cendl)

			for libxs, p in zip(cendl_xs, preds_cendl):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)
			evaluation_r2s.append(pred_cendl_r2)
		# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

		if nuc in JENDL_nuclides:
			jendl_test, jendl_xs = log_make_test(nuclides=[nuc], df=JENDL,
												 log_reduction_variable=log_reduction_var)

			log_preds_jendl = model.predict(jendl_test)
			jendl_xs = [np.exp(k) - log_reduction_var for k in jendl_xs]
			preds_jendl = [(np.exp(k) - log_reduction_var) for k in log_preds_jendl]

			pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=preds_jendl)

			for libxs, p in zip(jendl_xs, preds_jendl):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)
			evaluation_r2s.append(pred_jendl_r2)
		# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

		if nuc in JEFF_nuclides:
			jeff_test, jeff_xs = log_make_test(nuclides=[nuc], df=JEFF,
												 log_reduction_variable=log_reduction_var)
			log_preds_jeff = model.predict(jeff_test)
			jeff_xs = [np.exp(k) - log_reduction_var for k in jeff_xs]
			preds_jeff = [(np.exp(k) - log_reduction_var) for k in log_preds_jeff]

			pred_jeff_r2 = r2_score(y_true=jeff_xs, y_pred=preds_jeff)

			for libxs, p in zip(jeff_xs, preds_jeff):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)
			evaluation_r2s.append(pred_jeff_r2)

		if nuc in TENDL_nuclides:
			tendl_test, tendl_xs = log_make_test(nuclides=[nuc], df=TENDL,
												 log_reduction_variable=log_reduction_var)
			log_preds_tendl = model.predict(tendl_test)
			tendl_xs = [np.exp(k) - log_reduction_var for k in tendl_xs]
			preds_tendl = [(np.exp(k) - log_reduction_var) for k in log_preds_tendl]

			pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=preds_tendl)

			for libxs, p in zip(tendl_xs, preds_tendl):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)
			evaluation_r2s.append(pred_tendl_r2)
		# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

		mean_nuclide_r2 = np.mean(evaluation_r2s) # r2 for nuclide nuc

		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {mean_nuclide_r2:0.5f}")

		# mse = mean_squared_error(true_xs, pred_xs)

		# nuclide_mse.append([nuc[0], nuc[1], mse])
		nuclide_r2.append([nuc[0], nuc[1], mean_nuclide_r2])
		# individual_r2_list.append(r2)

	for val in y_test:
		every_true_value_list.append(val)
	# overall_r2_list.append(overall_r2)
	# print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
	# print(f"R2: {overall_r2}") # Total R^2 for all predictions in this training campaign
	time_taken = time.time() - time1
	print(f'completed in {time_taken} s.\n')


print()

# print(f"New overall r2: {r2_score(every_true_value_list, every_prediction_list)}")

all_libraries_r2 = r2_score(y_true=all_library_evaluations, y_pred= all_interpolated_predictions)
print(f"Overall R2: {all_libraries_r2:0.5f}")

A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]
# mse_log_plots = [np.log(i[-1]) for i in nuclide_mse]
# mse_plots = [i[-1] for i in nuclide_mse]

log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]

# heavy_performance = [abs(np.log(abs(i[-1]))) for i in nuclide_r2 if i[1] < 50]
# print(f"Light performance: {heavy_performance},\n mean: {np.mean(heavy_performance)}")

plt.figure()
plt.plot(A_plots, log_plots, 'x')
plt.xlabel("A")
plt.ylabel("log abs r2")
plt.title("log abs r2")
plt.grid()
plt.show()

plt.figure()
plt.plot(Z_plots, log_plots_Z, 'x')
plt.xlabel('Z')
plt.ylabel("log abs r2")
plt.title("log abs r2 - Z")
plt.grid()
plt.show()