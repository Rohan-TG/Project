import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import random
import xgboost as xg
import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import exp_make_test, exp_make_train, anomaly_remover, range_setter, delogger

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]


df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))


al = range_setter(la=30, ua=215, df=df)


validation_nuclides = [[44,103],[50,122], [82,208], [47,116], [40,90]]
validation_set_size = 30 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


if __name__ == "__main__":

	time1 = time.time()

	X_train, y_train = exp_make_train(df=df, validation_nuclides=validation_nuclides,la=30, ua=215) # make training matrix

	X_test, y_test = exp_make_test(validation_nuclides,df=df_test)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Data prep done")

	# model_seed = random.randint(a=1, b=1000)

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.01,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=42,)

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
				dummy_test_XS.append(np.log(y_test[i]))
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(np.log(predictions[i]))

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i] # validation nuclide
		plt.plot(erg, true_xs, label='data')
		plt.plot(erg, pred_xs, label='predictions', color='red')
		plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
		plt.legend()
		plt.grid()
		plt.ylabel('$\sigma_{n,2n}$ / b')
		plt.xlabel('Energy / MeV')
		plt.show()

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
		time.sleep(0.4)

	exp_true_xs = [np.log(y) for y in y_test]
	exp_pred_xs = [np.log(xs)for xs in predictions]

	print(f"MSE: {mean_squared_error(exp_true_xs, exp_pred_xs, squared=False)}") # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs):0.5f}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')
