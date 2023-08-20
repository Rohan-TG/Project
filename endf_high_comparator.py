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
from matrix_functions import make_test, make_train, anomaly_remover, range_setter, General_plotter

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]
df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=210, ua=260, df=df)

TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=210, ua=260)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=210, ua=260)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=210, ua=260)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=210, ua=260)

validation_nuclides = [[95,244]]
validation_set_size = 1 # number of nuclides hidden from training

# random.seed(a=42)

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=210, ua=260,) # make training matrix

X_test, y_test = make_test(validation_nuclides, df=df_test,)

# X_train must be in the shape (n_samples, n_features)
# and y_train must be in the shape (n_samples) of the target

print("Data prep done")

model = xg.XGBRegressor(n_estimators=900,
						learning_rate=0.008,
						max_depth=8,
						subsample=0.18236,
						max_leaves=0,
						seed=42,)

time1 = time.time()
model.fit(X_train, y_train)

print("Training complete")

predictions = model.predict(X_test) # XS predictions
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
			dummy_test_E.append(row[4]) # Energy values are in 5th row
			dummy_predictions.append(predictions[i])

	XS_plotmatrix.append(dummy_test_XS)
	E_plotmatrix.append(dummy_test_E)
	P_plotmatrix.append(dummy_predictions)

# plot predictions against data
# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):

	current_nuclide = validation_nuclides[i]

	jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
	cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
	jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
	tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])

	nuc = validation_nuclides[i] # validation nuclide
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
	plt.plot(tendlerg, tendlxs, label = "TENDL21", color='dimgrey', linewidth=2)
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF3.3',color='mediumvioletred')
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

	r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
	print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
	time.sleep(0.5)

print(f"MSE: {mean_squared_error(y_test, predictions, squared=False):0.5f}") # MSE
print(f"R2: {r2_score(y_test, predictions):0.5f}") # Total R^2 for all predictions in this training campaign
print(f'completed in {time.time() - time1:0.1f} s')


train_predictions = model.predict(X_train)

train_r2 = r2_score(y_true=y_train, y_pred=train_predictions)
train_mse = mean_squared_error(y_true=y_train, y_pred=train_predictions)
print(f"Training R2: {train_r2:0.5f}")
print(f"Training MSE: {train_mse:0.5f}")