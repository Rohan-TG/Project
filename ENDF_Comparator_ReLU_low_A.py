import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable
from sklearn.metrics import mean_squared_error
import scipy
from matrix_functions import General_plotter, range_setter, make_train, make_test
import random
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error
import time

TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=220, ua=260)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=220, ua=260)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=220, ua=260)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=220, ua=260)


df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df.index = range(len(df))
al = range_setter(la=220, ua=260, df=df)


validation_nuclides = []
validation_set_size = 5

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")



X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=220, ua=260,)
X_test, y_test = make_test(validation_nuclides, df=df)
print("Data prep done")

model = xg.XGBRegressor(n_estimators=500,
						learning_rate=0.01,
						max_depth=8,
						subsample=0.18236,
						max_leaves=0,
						seed=42, )

model.fit(X_train, y_train)
print("Training complete")
predictions = model.predict(X_test)  # XS predictions
predictions_ReLU = []
for pred in predictions:
	if pred >= 0.003:
		predictions_ReLU.append(pred)
	else:
		predictions_ReLU.append(0)

predictions = predictions_ReLU

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

for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):

	current_nuclide = validation_nuclides[i]

	jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
	cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
	jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
	tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])

	nuc = validation_nuclides[i]  # validation nuclide
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
	plt.plot(tendlerg, tendlxs, label="TENDL21", color='dimgrey', linewidth=2)
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF3.3', color='mediumvioletred')
	if nuc in JENDL_nuclides:
		plt.plot(jendlerg, jendlxs, label='JENDL5', color='green')
	if nuc in CENDL_nuclides:
		plt.plot(cendlerg, cendlxs, '--', label='CENDL3.2', color='gold')
	plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
	plt.legend()
	plt.grid()
	plt.ylabel('$\sigma_{n,2n}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)
	r2 = r2_score(true_xs, pred_xs)  # R^2 score for this specific nuclide
	print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}, MSE: {mse:0.5f}")
	time.sleep(0.8)

	# x_interpolate = np.linspace(stop=19.4, start=0, num=500)

	f_pred = scipy.interpolate.interp1d(x=erg, y=pred_xs)
	# predictions_interpolated = f_predictions(x_interpolate)

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


	# rmse_endf_tendl = mean_squared_error(endfb8_interp_xs, tendl_interp_xs)
	# rmse_endf_jendl = mean_squared_error(endfb8_interp_xs, jendl_interp_xs)
	# rmse_tendl_jendl = mean_squared_error(jendl_interp_xs, tendl_interp_xs)
	# rmse_tendl_jeff = mean_squared_error(jeff_interp_xs, tendl_interp_xs)
	# rmse_endfb_jeff = mean_squared_error(endfb8_interp_xs, jeff_interp_xs)
	# rmse_jeff_jendl = mean_squared_error(jeff_interp_xs, jendl_interp_xs)
	#
	# print(f"ENDF/B-VIII - TENDL21: {rmse_endf_tendl:0.6f}")
	# print(f"ENDF/B-VIII - JENDL5: {rmse_endf_jendl:0.6f}")
	# print(f"TENDL21 - JENDL5: {rmse_tendl_jendl:0.6f}")
	# print(f"TENDL21 - JEFF3.3: {rmse_tendl_jeff:0.6f}")
	# print(f"ENDF/B-VIII - JEFF3.3: {rmse_endfb_jeff:0.6f}")
	# print(f"JEFF3.3 - JENDL5: {rmse_jeff_jendl:0.6f}")

