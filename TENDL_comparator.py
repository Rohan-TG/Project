import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import random
import xgboost as xg
import time
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import make_test, make_train, anomaly_remover, range_setter, TENDL_plotter



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


validation_nuclides = [[22,49], [69,171], [81,203],
					   [73,181], [37,87], [73,180],
					   [80,197], [34,79], [42,99],
					   [18,38], [31,69], [56,139],
					   [16,33], [21,45], [61,146],
					   [30,64], [50,120], [51,126],
					   [20,47], [28,59], [54,136],
					   [44,103], [28,58], [52,130],
					   [52,128], [15,31], [39,91],
					   [34,82], ]
validation_set_size = 20 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")



if __name__ == "__main__":

	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=df_test)

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

	# shap_val_gpu = model.predict(X_test, pred_interactions=True)

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
				dummy_test_XS.append(y_test[i])
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

		if nuclide in TENDL_nuclides:
			dummy_tendl_XS = []
			dummy_tendl_ERG = []

			TENDL_ERG_matrix, TENDL_XS = TENDL_plotter(df=TENDL, nuclides=[nuclide])

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
		plt.plot(erg, pred_xs, label='Predictions')
		plt.plot(erg, true_xs, label='ENDF/B-VIII')
		plt.plot(tendl_erg_plot, tendl_xs_plot, label = "TENDL")
		plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
		plt.legend()
		plt.grid()
		plt.ylabel('XS / b')
		plt.xlabel('Energy / MeV')
		plt.show()

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
		time.sleep(0.5)

	print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
	print(f"R2: {r2_score(y_test, predictions)}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')
