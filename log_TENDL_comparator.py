import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import xgboost as xg
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


TENDL = pd.read_csv("TENDL_MT16_XS.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=215)

JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=215)

JENDL = pd.read_csv('JENDL4_features_arange_zeroed.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=215)

CENDL = pd.read_csv('CENDL33_features_arange_zeroed.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=215)


magic_numbers = [2, 8, 20, 28, 50, 82, 126]

doubly_magic = []
n_magic = []
p_magic = []
all_magic = []

for nuc in al:
	if nuc[0] in magic_numbers and (nuc[1] - nuc[0]) in magic_numbers:
		# print(f"Double: {nuc} - {periodictable.elements[nuc[0]]}-{nuc[1]}")
		doubly_magic.append(nuc)
		all_magic.append(nuc)
	elif nuc[0] in magic_numbers and (nuc[1] - nuc[0]) not in magic_numbers:
		# print(f"Protonic: {nuc} - {periodictable.elements[nuc[0]]}-{nuc[1]}")
		p_magic.append(nuc)
		all_magic.append(nuc)
	elif nuc[0] not in magic_numbers and (nuc[1] - nuc[0]) in magic_numbers:
		# print(f"Neutronic: {nuc} - {periodictable.elements[nuc[0]]}-{nuc[1]}")
		n_magic.append(nuc)
		all_magic.append(nuc)


validation_nuclides = [[38,87]]
validation_set_size = 20 # number of nuclides hidden from training

random.seed(a=42)

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")

log_reduction_var = 0.00001

if __name__ == "__main__":

	X_train, y_train = log_make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215,
									  log_reduction_variable=log_reduction_var) # make training matrix

	X_test, y_test = log_make_test(validation_nuclides, df=df_test,
								   log_reduction_variable=log_reduction_var)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Data prep done")

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.01,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=42,)

	time1 = time.time()
	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions

	# shap_val_gpu = model.predict(X_test, pred_interactions=True)

	# Form arrays for plots below
	XS_plotmatrix = []
	E_plotmatrix = []
	P_plotmatrix = []

	TENDL_XS_plotmatrix = []
	TENDL_E_plotmatrix = []

	JENDL_XS_plotmatrix = []
	JENDL_E_plotmatrix = []

	CENDL_XS_plotmatrix = []
	CENDL_E_plotmatrix = []

	JEFF_XS_plotmatrix = []
	JEFF_E_plotmatrix = []

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

		if nuclide in TENDL_nuclides:
			dummy_tendl_XS = []
			dummy_tendl_ERG = []

			TENDL_ERG_matrix, TENDL_XS = General_plotter(df=TENDL, nuclides=[nuclide])

			dummy_tendl_XS.append(TENDL_XS)
			dummy_tendl_ERG.append(TENDL_ERG_matrix[-1])

			TENDL_XS_plotmatrix.append(dummy_tendl_XS)
			TENDL_E_plotmatrix.append(dummy_tendl_ERG)
		else:
			TENDL_XS_plotmatrix.append([[]])
			TENDL_E_plotmatrix.append([[]])

		if nuclide in JEFF_nuclides:
			dummy_JEFF_XS = []
			dummy_JEFF_ERG = []

			JEFF_ERG_matrix, JEFF_XS = General_plotter(df=JEFF, nuclides=[nuclide])

			dummy_JEFF_XS.append(JEFF_XS)
			dummy_JEFF_ERG.append(JEFF_ERG_matrix[-1])

			JEFF_XS_plotmatrix.append(dummy_JEFF_XS)
			JEFF_E_plotmatrix.append(dummy_JEFF_ERG)
		else:
			JEFF_XS_plotmatrix.append([[]])
			JEFF_E_plotmatrix.append([[]])

		if nuclide in JENDL_nuclides:
			dummy_JENDL_XS = []
			dummy_JENDL_ERG = []

			JENDL_ERG_matrix, JENDL_XS = General_plotter(df=JENDL, nuclides=[nuclide])

			dummy_JENDL_XS.append(JENDL_XS)
			dummy_JENDL_ERG.append(JENDL_ERG_matrix[-1])

			JENDL_XS_plotmatrix.append(dummy_JENDL_XS)
			JENDL_E_plotmatrix.append(dummy_JENDL_ERG)
		else:
			JENDL_XS_plotmatrix.append([[]])
			JENDL_E_plotmatrix.append([[]])

		if nuclide in CENDL_nuclides:
			dummy_CENDL_XS = []
			dummy_CENDL_ERG = []

			CENDL_ERG_matrix, CENDL_XS = General_plotter(df=CENDL, nuclides=[nuclide])

			dummy_CENDL_XS.append(CENDL_XS)
			dummy_CENDL_ERG.append(CENDL_ERG_matrix[-1])

			CENDL_XS_plotmatrix.append(dummy_CENDL_XS)
			CENDL_E_plotmatrix.append(dummy_CENDL_ERG)
		else:
			CENDL_XS_plotmatrix.append([[]])
			CENDL_E_plotmatrix.append([[]])

			# print(TENDL_E_plotmatrix)

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg,
			tendl_xs, tendl_erg,
			jeffxs, jefferg,
			jendlxs, jendlerg,
			cendlxs, cendlerg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix,
												TENDL_XS_plotmatrix, TENDL_E_plotmatrix,
												JEFF_XS_plotmatrix, JEFF_E_plotmatrix,
												JENDL_XS_plotmatrix, JENDL_E_plotmatrix,
												CENDL_XS_plotmatrix, CENDL_E_plotmatrix)):

		current_nuclide = validation_nuclides[i]

		tendl_erg_plot = tendl_erg[0]
		tendl_xs_plot = tendl_xs[0]

		jefferg_plot = jefferg[0]
		jeffxs_plot = jeffxs[0]

		jendlerg_plot = jendlerg[0]
		jendlxs_plot = jendlxs[0]

		cendlerg_plot = cendlerg[0]
		cendlxs_plot = cendlxs[0]

		nuc = validation_nuclides[i] # validation nuclide
		plt.plot(erg, pred_xs, label='Predictions', color='red')
		plt.plot(erg, true_xs, label='ENDF/B-VIII')
		plt.plot(tendl_erg_plot, tendl_xs_plot, label = "TENDL21", color='dimgrey')
		if jefferg_plot != []:
			plt.plot(jefferg_plot, jeffxs_plot, label='JEFF3.3',color='mediumvioletred')
		if jendlerg_plot != []:
			plt.plot(jendlerg_plot, jendlxs_plot, label='JENDL4', color='green')
		if cendlerg_plot != []:
			plt.plot(cendlerg_plot, cendlxs_plot, label='CENDL3.3', color='gold')
		plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
		plt.legend()
		plt.grid()
		plt.ylabel('XS / b')
		plt.xlabel('Energy / MeV')
		plt.show()

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
		time.sleep(0.5)

	exp_true_xs = [np.exp(y) -log_reduction_var for y in y_test]
	exp_pred_xs = [np.exp(xs)- log_reduction_var for xs in predictions]

	print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs)}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')
