import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import xgboost as xg
import time
import shap
import scipy
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import log_make_test, log_make_train, anomaly_remover, range_setter, General_plotter

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

# df = pd.concat([df, df])

df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
al = range_setter(la=0, ua=50, df=df)


TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=50)

JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=50)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=50)

CENDL = pd.read_csv('CENDL32_features_arange_zeroed.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=50)


validation_nuclides = [[20,40]]
validation_set_size = 5 # number of nuclides hidden from training

# random.seed(a=42)

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")

log_reduction_var = 0.00001

if __name__ == "__main__":

	X_train, y_train = log_make_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=55,
									  log_reduction_variable=log_reduction_var) # make training matrix

	X_test, y_test = log_make_test(validation_nuclides, df=df_test,
								   log_reduction_variable=log_reduction_var)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Data prep done")

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.007,
							max_depth=8,
							subsample=0.5,
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


	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):

		current_nuclide = validation_nuclides[i]

		tendl_energy, tendl_xs = General_plotter(df=TENDL, nuclides=[current_nuclide])
		JEFF_energy, JEFF_XS = General_plotter(df=JEFF, nuclides=[current_nuclide])
		JENDL5_energy, JENDL5_XS = General_plotter(df=JENDL, nuclides=[current_nuclide])
		CENDL32_energy, CENDL32_XS = General_plotter(df=CENDL, nuclides=[current_nuclide])

		nuc = validation_nuclides[i] # validation nuclide
		plt.plot(erg, pred_xs, label='Predictions', color='red')
		plt.plot(erg, true_xs, label='ENDF/B-VIII')
		plt.plot(tendl_energy, tendl_xs, label = "TENDL21", color='dimgrey')
		if current_nuclide in JEFF_nuclides:
			plt.plot(JEFF_energy, JEFF_XS, label='JEFF3.3',color='mediumvioletred')
		if current_nuclide in JENDL_nuclides:
			plt.plot(JENDL5_energy, JENDL5_XS, label='JENDL5', color='green')
		if current_nuclide in CENDL_nuclides:
			plt.plot(CENDL32_energy, CENDL32_XS, label='CENDL3.2', color='gold')
		plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
		plt.legend()
		plt.grid()
		plt.ylabel('XS / b')
		plt.xlabel('Energy / MeV')
		plt.show()

		time.sleep(1)

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")

		if current_nuclide in al:
			endfb8_erg, endfb8_xs = General_plotter(df=df, nuclides=[current_nuclide])

			x_interpolate_endfb8 = np.linspace(start=0, stop=max(endfb8_erg), num=500)
			f_pred_endfb8 = scipy.interpolate.interp1d(x=erg, y=pred_xs)
			predictions_interpolated_endfb8 = f_pred_endfb8(x_interpolate_endfb8)

			f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg, y=endfb8_xs)
			endfb8_interp_xs = f_endfb8(x_interpolate_endfb8)
			pred_endfb_mse = mean_squared_error(predictions_interpolated_endfb8, endfb8_interp_xs)
			pred_endfb_r2 = r2_score(y_true=endfb8_interp_xs, y_pred=predictions_interpolated_endfb8)
			print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

		if current_nuclide in CENDL_nuclides:
			cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[current_nuclide])

			x_interpolate_cendl = np.linspace(start=0, stop=max(cendl_erg), num=500)
			f_pred_cendl = scipy.interpolate.interp1d(x=erg, y=pred_xs)
			predictions_interpolated_cendl = f_pred_cendl(x_interpolate_cendl)

			f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs)
			cendl_interp_xs = f_cendl32(x_interpolate_cendl)
			pred_cendl_mse = mean_squared_error(predictions_interpolated_cendl, cendl_interp_xs)
			pred_cendl_r2 = r2_score(y_true=cendl_interp_xs, y_pred=predictions_interpolated_cendl)
			print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

		if current_nuclide in JENDL_nuclides:
			jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[current_nuclide])

			x_interpolate_jendl = np.linspace(start=0, stop=max(jendl_erg), num=500)
			f_pred_jendl = scipy.interpolate.interp1d(x=erg, y=pred_xs)
			predictions_interpolated_jendl = f_pred_jendl(x_interpolate_jendl)

			f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs)
			jendl_interp_xs = f_jendl5(x_interpolate_jendl)
			pred_jendl_mse = mean_squared_error(predictions_interpolated_jendl, jendl_interp_xs)
			pred_jendl_r2 = r2_score(y_true=jendl_interp_xs, y_pred=predictions_interpolated_jendl)
			print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

		if current_nuclide in JEFF_nuclides:
			jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[current_nuclide])

			x_interpolate_jeff = np.linspace(start=0, stop=max(jeff_erg), num=500)
			f_pred_jeff = scipy.interpolate.interp1d(x=erg, y=pred_xs)
			predictions_interpolated_jeff = f_pred_jeff(x_interpolate_jeff)

			f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs)
			jeff_interp_xs = f_jeff33(x_interpolate_jeff)
			pred_jeff_mse = mean_squared_error(predictions_interpolated_jeff, jeff_interp_xs)
			pred_jeff_r2 = r2_score(y_true=jeff_interp_xs, y_pred=predictions_interpolated_jeff)
			print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

		if current_nuclide in TENDL_nuclides:
			tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[current_nuclide])

			x_interpolate_tendl = np.linspace(start=0, stop=max(tendl_erg), num=500)
			f_pred_tendl = scipy.interpolate.interp1d(x=erg, y=pred_xs)
			predictions_interpolated_tendl = f_pred_tendl(x_interpolate_tendl)

			f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs)
			tendl_interp_xs = f_tendl21(x_interpolate_tendl)
			pred_tendl_mse = mean_squared_error(predictions_interpolated_tendl, tendl_interp_xs)
			pred_tendl_r2 = r2_score(y_true=tendl_interp_xs, y_pred=predictions_interpolated_tendl)
			print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

	exp_true_xs = [np.exp(y) -log_reduction_var for y in y_test]
	exp_pred_xs = [np.exp(xs)- log_reduction_var for xs in predictions]

	print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs)}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')
