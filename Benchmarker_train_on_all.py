import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
# import shap
import scipy
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import make_train, make_test, range_setter, General_plotter

endfb8 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
jendl5 = pd.read_csv('JENDL5_arange_all_features.csv')
tendl21 = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
jeff33 = pd.read_csv('JEFF33_features_arange_zeroed.csv')
cendl32 = pd.read_csv('CENDL32_features_arange_zeroed.csv')


CENDL_nuclides = range_setter(df=cendl32, la=30, ua=210)
JENDL_nuclides = range_setter(df=jendl5, la=30, ua=210)
JEFF_nuclides = range_setter(df=jeff33, la=30, ua=210)
TENDL_nuclides = range_setter(df=tendl21, la=30, ua=210)
ENDFB_nuclides = range_setter(df=endfb8, la=30, ua=210)

all_libraries = pd.concat([endfb8, jendl5])

# all_libraries = pd.concat([all_libraries, tendl21])
all_libraries = pd.concat([all_libraries, jeff33])
all_libraries = pd.concat([all_libraries, cendl32])

nucs = range_setter(df=all_libraries, la=30, ua=210)

all_libraries.index = range(len(all_libraries))


al = range_setter(la=30, ua=210, df=all_libraries)


nuclides_used = []

nuclide_mse = []
nuclide_r2 = []


# overall_r2_list = []

every_prediction_list = []
every_true_value_list = []

all_library_evaluations = []
all_interpolated_predictions = []

counter = 0
while len(nuclides_used) < len(ENDFB_nuclides):

	# print(f"{len(nuclides_used) // len(al)} Epochs left"

	time1 = time.time()
	validation_nuclides = []  # list of nuclides used for validation
	validation_set_size = 20  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:

		choice = random.choice(ENDFB_nuclides)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides and choice not in nuclides_used:
			validation_nuclides.append(choice)
			nuclides_used.append(choice)
		if len(nuclides_used) == len(ENDFB_nuclides):
			break


	print("Test nuclide selection complete")
	print(f"{len(nuclides_used)}/{len(ENDFB_nuclides)} selections")


	X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides, la=30, ua=210) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=tendl21)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Train/val matrices generated")


	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.015,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=42,)

	# model = xg.XGBRegressor(n_estimators=900,  this is the new one
	# 						reg_lambda=0.9530201929544536,
	# 						learning_rate=0.005,
	# 						max_depth=12,
	# 						subsample=0.15315395901090592)


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
		# dummy_shell = []
		for i, row in enumerate(X_test):
			if [row[0], row[1]] == nuclide:
				dummy_test_XS.append(y_test[i])
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(predictions[i])
				# dummy_shell.append(row[2])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	# plot predictions against data
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i]
		current_nuclide = nuc

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")

		mse = mean_squared_error(true_xs, pred_xs)

		nuclide_mse.append([nuc[0], nuc[1], mse])
		nuclide_r2.append([nuc[0], nuc[1], r2])

		f_pred = scipy.interpolate.interp1d(x=erg, y=pred_xs)

		pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
		pred_endfb_r2 = r2_score(y_true=true_xs, y_pred=pred_xs)
		for libxs, p in zip(true_xs, pred_xs):
			all_library_evaluations.append(libxs)
			all_interpolated_predictions.append(p)

		print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

		if current_nuclide in CENDL_nuclides:
			cendl_erg, cendl_xs = General_plotter(df=cendl32, nuclides=[current_nuclide])

			predictions_interpolated_cendl = f_pred(cendl_erg)

			pred_cendl_mse = mean_squared_error(predictions_interpolated_cendl, cendl_xs)
			pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=predictions_interpolated_cendl)
			print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")
			for libxs, p in zip(cendl_xs[1:], predictions_interpolated_cendl[1:]):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)


		if current_nuclide in JENDL_nuclides:
			jendl_erg, jendl_xs = General_plotter(df=jendl5, nuclides=[current_nuclide])

			predictions_interpolated_jendl = f_pred(jendl_erg)

			pred_jendl_mse = mean_squared_error(predictions_interpolated_jendl, jendl_xs)
			pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=predictions_interpolated_jendl)
			print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")
			for libxs, p in zip(jendl_xs[1:], predictions_interpolated_jendl[1:]):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)

		if current_nuclide in JEFF_nuclides:
			jeff_erg, jeff_xs = General_plotter(df=jeff33, nuclides=[current_nuclide])

			predictions_interpolated_jeff = f_pred(jeff_erg)

			pred_jeff_mse = mean_squared_error(predictions_interpolated_jeff, jeff_xs)
			pred_jeff_r2 = r2_score(y_true=jeff_xs, y_pred=predictions_interpolated_jeff)
			print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")
			for libxs, p in zip(jeff_xs[1:], predictions_interpolated_jeff[1:]):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)

		if current_nuclide in TENDL_nuclides:
			tendl_erg, tendl_xs = General_plotter(df=tendl21, nuclides=[current_nuclide])
			predictions_interpolated_tendl = f_pred(tendl_erg)

			pred_tendl_mse = mean_squared_error(predictions_interpolated_tendl, tendl_xs)
			pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=predictions_interpolated_tendl)
			print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")
			for libxs, p in zip(tendl_xs[1:], predictions_interpolated_tendl[1:]):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)

	overall_r2 = r2_score(y_test, predictions)
	for pred in predictions:
		every_prediction_list.append(pred)

	for val in y_test:
		every_true_value_list.append(val)
	# overall_r2_list.append(overall_r2)
	# print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
	print(f"R2: {overall_r2}") # Total R^2 for all predictions in this training campaign
	time_taken = float(time.time() - time1)
	print(f'completed in {time_taken:0.1f} s.')
	counter += 1

	time_remaining = ((len(al) - counter*validation_set_size) / validation_set_size) * time_taken
	print(f"Time remaining: {time_remaining:0.1f} s. \n")


print()



A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]
all_libraries_mse = mean_squared_error(y_true=all_library_evaluations, y_pred=all_interpolated_predictions)
all_libraries_r2 = r2_score(y_true=all_library_evaluations, y_pred= all_interpolated_predictions)
print(f"MSE: {all_libraries_mse:0.5f}")
print(f"R2: {all_libraries_r2:0.5f}")

log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]


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


print(f"Overall r2: {r2_score(every_true_value_list, every_prediction_list):0.5f}")
