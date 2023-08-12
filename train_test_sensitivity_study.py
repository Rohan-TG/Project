import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import tqdm
import time
# import shap
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
import scipy
from matrix_functions import make_train, make_test, range_setter, General_plotter

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")  # dataframe as above, but with the new features from the Gilbert-Cameron model
al = range_setter(df=df, la=30, ua=210)

TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)

JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)

CENDL = pd.read_csv('CENDL32_features_arange_zeroed.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)



nuclides_used = []
nuclide_mse = []
nuclide_r2 = []


every_prediction_list = []
every_true_value_list = []




val_set_sizes = [5, 10, 20, 30, 40, 50, 75, 80, 90, 100]
# val_set_sizes = [200, 250]
set_overall_r2 = []
set_overall_train_r2 = []
set_overall_mse = []


for set in tqdm.tqdm(val_set_sizes):
	individual_r2_list = []

	all_library_evaluations = []
	all_interpolated_predictions = []
	while len(nuclides_used) < len(al):


		validation_nuclides = []  # list of nuclides used for validation
		validation_set_size = set  # number of nuclides hidden from training
		overall_r2_list = []

		while len(validation_nuclides) < validation_set_size:

			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
			if len(nuclides_used) == len(al):
				break


		print("Test nuclide selection complete")
		print(f"{len(nuclides_used)}/{len(al)} selections")
		# print(f"Epoch {len(al) // len(nuclides_used) + 1}/")


		X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=210) # make training matrix

		X_test, y_test = make_test(validation_nuclides, df=df_test)

		print("Train/val matrices generated")

		model = xg.XGBRegressor(n_estimators=900,
								learning_rate=0.008,
								max_depth=8,
								subsample=0.18236,
								max_leaves=0,
								seed=42,)

		# model = xg.XGBRegressor(n_estimators=600,
		# 						reg_lambda=1.959,
		# 						learning_rate=0.00856,
		# 						max_depth=7,
		# 						subsample=0.1895,
		# 						seed=42)

		time1 = time.time()
		model.fit(X_train, y_train)

		print("Training complete")

		predictions = model.predict(X_test) # XS predictions
		predictions_ReLU = []
		for pred in predictions:
			if pred >= 0.001:
				predictions_ReLU.append(pred)
			else:
				predictions_ReLU.append(0)

		predictions = predictions_ReLU

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
		for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
			nuc = validation_nuclides[i]
			current_nuclide = nuc

			evaluation_r2s = []

			f_pred = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')

			for libxs, p in zip(true_xs, pred_xs):
				all_library_evaluations.append(libxs)
				all_interpolated_predictions.append(p)

			if current_nuclide in CENDL_nuclides:
				cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[current_nuclide])

				predictions_interpolated_cendl = f_pred(cendl_erg)

				pred_cendl_mse = mean_squared_error(predictions_interpolated_cendl[1:], cendl_xs[1:])
				pred_cendl_r2 = r2_score(y_true=cendl_xs[1:], y_pred=predictions_interpolated_cendl[1:])

				for libxs, p in zip(cendl_xs[1:], predictions_interpolated_cendl[1:]):
					all_library_evaluations.append(libxs)
					all_interpolated_predictions.append(p)
				evaluation_r2s.append(pred_cendl_r2)
			# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

			if current_nuclide in JENDL_nuclides:
				jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[current_nuclide])

				predictions_interpolated_jendl = f_pred(jendl_erg)

				pred_jendl_mse = mean_squared_error(predictions_interpolated_jendl[1:], jendl_xs[1:])
				pred_jendl_r2 = r2_score(y_true=jendl_xs[1:], y_pred=predictions_interpolated_jendl[1:])

				for libxs, p in zip(jendl_xs[1:], predictions_interpolated_jendl[1:]):
					all_library_evaluations.append(libxs)
					all_interpolated_predictions.append(p)
				evaluation_r2s.append(pred_jendl_r2)
			# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

			if current_nuclide in JEFF_nuclides:
				jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[current_nuclide])

				predictions_interpolated_jeff = f_pred(jeff_erg)

				pred_jeff_mse = mean_squared_error(predictions_interpolated_jeff[1:], jeff_xs[1:])
				pred_jeff_r2 = r2_score(y_true=jeff_xs[1:], y_pred=predictions_interpolated_jeff[1:])

				for libxs, p in zip(jeff_xs[1:], predictions_interpolated_jeff[1:]):
					all_library_evaluations.append(libxs)
					all_interpolated_predictions.append(p)
				evaluation_r2s.append(pred_jeff_r2)
			# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

			if current_nuclide in TENDL_nuclides:
				tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[current_nuclide])
				predictions_interpolated_tendl = f_pred(tendl_erg)

				pred_tendl_mse = mean_squared_error(predictions_interpolated_tendl[1:], tendl_xs[1:])
				pred_tendl_r2 = r2_score(y_true=tendl_xs[1:], y_pred=predictions_interpolated_tendl[1:])

				for libxs, p in zip(tendl_xs[1:], predictions_interpolated_tendl[1:]):
					all_library_evaluations.append(libxs)
					all_interpolated_predictions.append(p)
				evaluation_r2s.append(pred_tendl_r2)

			r2 = np.mean(evaluation_r2s)
			# r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
			print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")

			mse = mean_squared_error(true_xs, pred_xs)

			nuclide_mse.append([nuc[0], nuc[1], mse])
			nuclide_r2.append([nuc[0], nuc[1], r2])
			individual_r2_list.append(r2)


		training_predictions = model.predict(X_train)
		tr_pred_return = []
		for p in training_predictions:
			tr_pred_return.append(p)

		overall_r2 = r2_score(y_test, predictions)
		for pred in predictions:
			every_prediction_list.append(pred)

		for val in y_test:
			every_true_value_list.append(val)
		overall_r2_list.append(overall_r2)
		# print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
		# print(f"R2: {overall_r2}") # Total R^2 for all predictions in this training campaign
		time_taken = time.time() - time1
		print(f'completed in {time_taken:0.1f} s.\n')


	# batch_r2 = r2_score(every_true_value_list, every_prediction_list)
	# # batch_r2 = np.mean(individual_r2_list)
	#
	batch_train_r2 = r2_score(tr_pred_return, y_train)

	all_libraries_mse = mean_squared_error(y_true=all_library_evaluations, y_pred=all_interpolated_predictions)
	all_libraries_r2 = r2_score(y_true=all_library_evaluations, y_pred=all_interpolated_predictions)
	print(f"MSE: {all_libraries_mse:0.5f}")
	print(f"R2: {all_libraries_r2:0.5f}")

	set_overall_r2.append(all_libraries_r2)
	set_overall_mse.append(all_libraries_mse)

	set_overall_train_r2.append(batch_train_r2)

	nuclides_used = []



print()

for set, r2, r2train in zip(val_set_sizes, set_overall_r2, set_overall_train_r2):
	print(f"Set size: {set} Test R2: {r2:0.5f} Train R2: {set_overall_train_r2}")