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
from matrix_functions import make_train, make_test, range_setter, r2_standardiser

endfb8 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
jendl5 = pd.read_csv('JENDL5_arange_all_features.csv')
tendl21 = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
jeff33 = pd.read_csv('JEFF33_all_features.csv')
cendl32 = pd.read_csv('CENDL32_all_features.csv')


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

benchmark_total_library_evaluations = []
benchmark_total_predictions = []


validation_set_size = 20
counter = 0
while len(nuclides_used) < len(al):

	# print(f"{len(nuclides_used) // len(al)} Epochs left")

	validation_nuclides = []  # list of nuclides used for validation

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

	X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides, la=0, ua=100)  # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=all_libraries)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Train/val matrices generated")


	model = xg.XGBRegressor(n_estimators=1200,
							learning_rate=0.004,
							max_depth=11,
							subsample=0.1556,
							max_leaves=0,
							seed=42,)


	model.fit(X_train, y_train)

	print("Training complete")

	print("Training complete")

	predictions = model.predict(X_test)  # XS predictions
	predictions_ReLU = []
	for pred in predictions:
		if pred >= 0.003:
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
				dummy_test_E.append(row[4])  # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	# plot predictions against data
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i]
		current_nuclide = nuc

		evaluation_r2s = []

		truncated_library_r2 = []

		all_library_evaluations = []
		all_predictions = []

		limited_evaluations = []
		limited_predictions = []

		try:
			pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(raw_predictions=pred_xs,
																		  library_xs=true_xs)
		except:
			print(current_nuclide)
		for libxs, p in zip(truncated_endfb, pred_endfb_gated):
			all_library_evaluations.append(libxs)
			all_predictions.append(p)

			benchmark_total_library_evaluations.append(libxs)
			benchmark_total_predictions.append(p)

		evaluation_r2s.append(endfb_r2)

		if current_nuclide in CENDL_nuclides:

			cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=cendl32)
			pred_cendl = model.predict(cendl_test)

			pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
			pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl,
																			   library_xs=cendl_xs)
			for libxs, p in zip(truncated_cendl, pred_cendl_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)

				benchmark_total_library_evaluations.append(libxs)
				benchmark_total_predictions.append(p)

			evaluation_r2s.append(pred_cendl_r2)
			truncated_library_r2.append(pred_cendl_r2)
		# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

		if current_nuclide in JENDL_nuclides:
			jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=jendl5)
			pred_jendl = model.predict(jendl_test)

			pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
			pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl,
																			   library_xs=jendl_xs)
			for libxs, p in zip(truncated_jendl, pred_jendl_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)

				benchmark_total_library_evaluations.append(libxs)
				benchmark_total_predictions.append(p)
			evaluation_r2s.append(pred_jendl_r2)
			truncated_library_r2.append(pred_jendl_r2)
		# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

		if current_nuclide in JEFF_nuclides:
			jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=jeff33)

			pred_jeff = model.predict(jeff_test)

			pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
			pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff,
																			library_xs=jeff_xs)
			for libxs, p in zip(truncated_jeff, pred_jeff_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)

				benchmark_total_library_evaluations.append(libxs)
				benchmark_total_predictions.append(p)

			evaluation_r2s.append(pred_jeff_r2)
			truncated_library_r2.append(pred_jeff_r2)
		# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

		if current_nuclide in TENDL_nuclides:
			tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=tendl21)
			pred_tendl = model.predict(tendl_test)

			pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
			pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl,
																			   library_xs=tendl_xs)
			for libxs, p in zip(truncated_tendl, pred_tendl_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)

				benchmark_total_library_evaluations.append(libxs)
				benchmark_total_predictions.append(p)

			evaluation_r2s.append(pred_tendl_r2)
			truncated_library_r2.append(pred_tendl_r2)
		# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}"

		r2 = r2_score(all_library_evaluations, all_predictions)  # various comparisons

		limited_r2 = r2_score(limited_evaluations, limited_predictions)


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
