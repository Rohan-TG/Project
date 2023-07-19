import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import random
import xgboost as xg
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import log_make_test, log_make_train, anomaly_remover, range_setter, General_plotter

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]
df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=30, ua=215, df=df)


TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=215)


# validation_nuclides = [[39,85], [39,86],[39,87],
# 					   [39,88],[39,90],[39,91],
# 					   [39,92],[39,93],[39,94],
# 					   [39,95]]

validation_nuclides = [[40,85], [40,86], [40,87],
					   [40,88], [40,89], [40,93],
					   [40,94], [40,95], [40,96],
					   [40,97], [40,98], [40,99],
					   [40,100], [40,101], [40,102],
					   ]

validation_set_size = 1 # number of nuclides hidden from training

lrv = 0.00001

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")



if __name__ == "__main__":

	X_train, y_train = log_make_train(df=df, validation_nuclides=validation_nuclides,
									  la=30, ua=215, log_reduction_variable=lrv) # make training matrix

	X_test, y_test = log_make_test(validation_nuclides, df=TENDL,
								   log_reduction_variable=lrv)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Data prep done")

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.015,
							max_depth=8,
							subsample=0.18236,
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

	TENDL_XS_plotmatrix = []
	TENDL_E_plotmatrix = []

	for nuclide in validation_nuclides:
		dummy_test_XS = []
		dummy_test_E = []
		dummy_predictions = []
		for i, row in enumerate(X_test):
			if [row[0], row[1]] == nuclide:
				dummy_test_XS.append(np.exp(y_test[i]) - lrv)
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(np.exp(predictions[i]) - lrv)

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

		if nuclide in TENDL_nuclides:
			dummy_tendl_XS = []
			dummy_tendl_ERG = []

			TENDL_ERG_matrix, TENDL_XS = General_plotter(df=TENDL, nuclides=[nuclide])

			dummy_tendl_XS.append(TENDL_XS)
			dummy_tendl_ERG.append(TENDL_ERG_matrix[-1])

			TENDL_XS_plotmatrix.append(dummy_tendl_XS)
			TENDL_E_plotmatrix.append(dummy_tendl_ERG)

			# print(TENDL_E_plotmatrix)

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg, tendl_xs, tendl_erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix, TENDL_XS_plotmatrix, TENDL_E_plotmatrix)):
		tendl_erg_plot = tendl_erg[0]
		tendl_xs_plot = tendl_xs[0]
		nuc = validation_nuclides[i] # validation nuclide
		plt.plot(erg, true_xs, label='ENDF/B-VIII')
		plt.plot(erg, pred_xs, label='Predictions')
		plt.plot(tendl_erg_plot, tendl_xs_plot, label = "TENDL21")
		plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
		plt.legend()
		plt.grid()
		plt.ylabel('XS / b')
		plt.xlabel('Energy / MeV')
		plt.show()

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
		time.sleep(0.5)


	exp_true_xs = [np.exp(y) - lrv for y in y_test]
	exp_pred_xs = [np.exp(xs)- lrv for xs in predictions]

	print(f"MSE: {mean_squared_error(exp_true_xs, exp_pred_xs, squared=False)}") # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs)}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')
