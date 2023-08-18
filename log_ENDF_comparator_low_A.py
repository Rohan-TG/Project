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
al = range_setter(la=0, ua=60, df=df)


TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=60)

JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=60)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=60)

CENDL = pd.read_csv('CENDL32_features_arange_zeroed.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=60)


validation_nuclides = [[20,40]]
validation_set_size = 15 # number of nuclides hidden from training

# random.seed(a=42)

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")

log_reduction_var = 0.00001


X_train, y_train = log_make_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=60,
								  log_reduction_variable=log_reduction_var) # make training matrix

X_test, y_test = log_make_test(validation_nuclides, df=df_test,
							   log_reduction_variable=log_reduction_var)

# X_train must be in the shape (n_samples, n_features)
# and y_train must be in the shape (n_samples) of the target

print("Data prep done")

model = xg.XGBRegressor(n_estimators=600,
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

	f_pred = scipy.interpolate.interp1d(x=erg, y=pred_xs)

	if current_nuclide in CENDL_nuclides:
		cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[current_nuclide])

		predictions_interpolated_cendl = f_pred(cendl_erg)

		pred_cendl_mse = mean_squared_error(predictions_interpolated_cendl[1:], cendl_xs[1:])
		pred_cendl_r2 = r2_score(y_true=cendl_xs[1:], y_pred=predictions_interpolated_cendl[1:])
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if current_nuclide in JENDL_nuclides:
		jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[current_nuclide])

		predictions_interpolated_jendl = f_pred(jendl_erg)

		pred_jendl_mse = mean_squared_error(predictions_interpolated_jendl[1:], jendl_xs[1:])
		pred_jendl_r2 = r2_score(y_true=jendl_xs[1:], y_pred=predictions_interpolated_jendl[1:])
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if current_nuclide in JEFF_nuclides:
		jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[current_nuclide])

		predictions_interpolated_jeff = f_pred(jeff_erg)

		pred_jeff_mse = mean_squared_error(predictions_interpolated_jeff[1:], jeff_xs[1:])
		pred_jeff_r2 = r2_score(y_true=jeff_xs[1:], y_pred=predictions_interpolated_jeff[1:])
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if current_nuclide in TENDL_nuclides:
		tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[current_nuclide])
		predictions_interpolated_tendl = f_pred(tendl_erg)

		pred_tendl_mse = mean_squared_error(predictions_interpolated_tendl[1:], tendl_xs[1:])
		pred_tendl_r2 = r2_score(y_true=tendl_xs[1:], y_pred=predictions_interpolated_tendl[1:])
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

exp_true_xs = [np.exp(y) -log_reduction_var for y in y_test]
exp_pred_xs = [np.exp(xs)- log_reduction_var for xs in predictions]

print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
print(f"R2: {r2_score(exp_true_xs, exp_pred_xs)}") # Total R^2 for all predictions in this training campaign
print(f'completed in {time.time() - time1} s')
