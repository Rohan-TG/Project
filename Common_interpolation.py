import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable
from sklearn.metrics import mean_squared_error
import scipy
from matrix_functions import General_plotter, range_setter, anomaly_remover, make_train, make_test
import random
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error
import time

TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=215)

JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=215)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=215)

CENDL = pd.read_csv('CENDL33_features_arange_zeroed.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=215)


df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]
df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=30, ua=215, df=df)


validation_nuclides = []
validation_set_size = 20

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")



X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215,)
X_test, y_test = make_test(validation_nuclides, df=df_test,)
print("Data prep done")

model = xg.XGBRegressor(n_estimators=900,
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
	if pred >= 0.001:
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

	x_interpolate = np.linspace(stop=19.6, start=0, num=500)

	f_predictions = scipy.interpolate.interp1d(x=erg, y=pred_xs)
	predictions_interpolated = f_predictions(x_interpolate)


	if current_nuclide in al:
		endfb8_erg, endfb8_xs = General_plotter(df=df, nuclides=[current_nuclide])
		f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg, y=endfb8_xs)
		endfb8_interp_xs = f_endfb8(x_interpolate)
		pred_endfb_mse = mean_squared_error(predictions_interpolated, endfb8_interp_xs)
		pred_endfb_r2 = r2_score(y_true=endfb8_interp_xs, y_pred=predictions_interpolated)
		print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

	if current_nuclide in CENDL_nuclides:
		cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[current_nuclide])
		f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs)
		cendl_interp_xs = f_cendl32(x_interpolate)
		pred_cendl_mse = mean_squared_error(predictions_interpolated, cendl_interp_xs)
		pred_cendl_r2 = r2_score(y_true=cendl_interp_xs, y_pred=predictions_interpolated)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if current_nuclide in JENDL_nuclides and max(erg) <= 18:
		jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[current_nuclide])
		f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs)
		jendl_interp_xs = f_jendl5(x_interpolate)
		pred_jendl_mse = mean_squared_error(predictions_interpolated, jendl_interp_xs)
		pred_jendl_r2 = r2_score(y_true=jendl_interp_xs, y_pred=predictions_interpolated)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if current_nuclide in JEFF_nuclides:
		jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[current_nuclide])
		f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs)
		jeff_interp_xs = f_jeff33(x_interpolate)
		pred_jeff_mse = mean_squared_error(predictions_interpolated, jeff_interp_xs)
		pred_jeff_r2 = r2_score(y_true=jeff_interp_xs, y_pred=predictions_interpolated)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if current_nuclide in TENDL_nuclides:
		tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[current_nuclide])
		f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs)
		tendl_interp_xs = f_tendl21(x_interpolate)
		pred_tendl_mse = mean_squared_error(predictions_interpolated, tendl_interp_xs)
		pred_tendl_r2 = r2_score(y_true=tendl_interp_xs, y_pred=predictions_interpolated)
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")



	# print(f"ENDF/B-VIII - TENDL21: {rmse_endf_tendl:0.6f}")
	# print(f"ENDF/B-VIII - JENDL5: {rmse_endf_jendl:0.6f}")
	# print(f"TENDL21 - JENDL5: {rmse_tendl_jendl:0.6f}")
	# print(f"TENDL21 - JEFF3.3: {rmse_tendl_jeff:0.6f}")
	# print(f"ENDF/B-VIII - JEFF3.3: {rmse_endfb_jeff:0.6f}")
	# print(f"JEFF3.3 - JENDL5: {rmse_jeff_jendl:0.6f}")

	# rmse_endf_tendl = mean_squared_error(endfb8_interp_xs, tendl_interp_xs)
	# rmse_endf_jendl = mean_squared_error(endfb8_interp_xs, jendl_interp_xs)
	# rmse_tendl_jendl = mean_squared_error(jendl_interp_xs, tendl_interp_xs)
	# rmse_tendl_jeff = mean_squared_error(jeff_interp_xs, tendl_interp_xs)
	# rmse_endfb_jeff = mean_squared_error(endfb8_interp_xs, jeff_interp_xs)
	# rmse_jeff_jendl = mean_squared_error(jeff_interp_xs, jendl_interp_xs)