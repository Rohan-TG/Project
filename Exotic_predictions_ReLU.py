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
from matrix_functions import make_test, make_train, r2_standardiser, range_setter, General_plotter, dsigma_dE

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df = df[df.Z != 11]
df.index = range(len(df))
al = range_setter(la=30, ua=215, df=df)


TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)
print("Data loaded")

# validation_nuclides = [[39,85], [39,86],[39,87],
# 					   [39,88],[39,90],[39,91],
# 					   [39,92],[39,93],[39,94],
# 					   [39,95]]

# validation_nuclides = [[40,86],[40,87],
# 					   # [40,88],[40,89],[40,94],
# 					   # [40,95],[40,96],[40,97],
# 					   # [40,98],[40,99],[40,100],
# 					   [40,101],[40,102],[78,188],
# 					   [49,119],]

validation_nuclides = []

proton_excess = 0
neutron_excess = 0

all_neutron_deficients = []
all_proton_deficients = []
mass_difference_with_r2 = []

exotic_nuclides = []
for x in TENDL_nuclides:
	if x not in al:
		exotic_nuclides.append(x)

# validation_nuclides = [[77,190],[74,175], [60,151], [78,188],
# 					   [49,110], [58,150], [49,119],
# 					   [55,127], [67,155], [71,172]]
# validation_nuclides = []

validation_set_size = 25

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(exotic_nuclides) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215) # make training matrix

X_test, y_test = make_test(validation_nuclides, df=TENDL)

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

TENDL_XS_plotmatrix = []
TENDL_E_plotmatrix = []

JENDL_XS_plotmatrix = []
JENDL_E_plotmatrix = []

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

	all_libs = []
	all_preds = []

	nuc = validation_nuclides[i] # validation nuclide

	jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[nuc])
	tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[nuc])

	if nuc in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[nuc], df=JENDL)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl, library_xs=jendl_xs)

		for p, xs in zip(pred_jendl_gated, truncated_jendl):
			all_libs.append(xs)
			all_preds.append(p)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")

	if nuc in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[nuc], df=TENDL)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl, library_xs=tendl_xs)

		for p, xs in zip(pred_tendl_gated, truncated_tendl):
			all_libs.append(xs)
			all_preds.append(p)
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {pred_tendl_r2:0.5f}")

	match_element = []
	for nuclide in al:
		if nuclide[0] == nuc[0]:
			match_element.append(nuclide)
	nuc_differences = []
	for match_nuclide in match_element:
		difference = match_nuclide[1] - nuc[1]
		nuc_differences.append(difference)

	if len(nuc_differences) > 0:
		if nuc_differences[0] > 0:
			min_difference = min(nuc_differences)
			proton_excess += 1
			all_neutron_deficients.append(pred_tendl_r2)
		elif nuc_differences[0] < 0:
			min_difference = max(nuc_differences)
			neutron_excess += 1

			all_proton_deficients.append(pred_tendl_r2)

		mass_difference_with_r2.append([pred_tendl_r2, min_difference])
		print(f"Mass difference: {min_difference:0.0f}\n")

	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(tendl_erg, tendl_xs, label = "TENDL21", color='dimgrey')
	if len(jendl_erg) > 1:
		plt.plot(jendl_erg, jendl_xs, label ='JENDL5', color='green')
	if nuc in al:
		plt.plot(erg, true_xs, label = 'ENDF/B-VIII')
	plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
	plt.legend()
	plt.grid()
	plt.ylabel('$\sigma_{n,2n}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	# r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide

	time.sleep(0.5)

	# print(f"Turning points: {dsigma_dE(XS=pred_xs)}\n")

print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
print(f"R2: {r2_score(y_test, predictions)}") # Total R^2 for all predictions in this training campaign
print(f'completed in {time.time() - time1} s')
