import pandas as pd
import random
import matplotlib.pyplot as plt
import periodictable
import xgboost as xg
from matrix_functions import make_test, make_train, range_setter, General_plotter
from sklearn.metrics import r2_score, mean_squared_error
import time

endfb8 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
jendl5 = pd.read_csv('JENDL5_arange_all_features.csv')
tendl21 = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
jeff33 = pd.read_csv('JEFF33_features_arange_zeroed.csv')
cendl32 = pd.read_csv('CENDL33_features_arange_zeroed.csv')

all_libraries = pd.concat([endfb8, jendl5])

# all_libraries = pd.concat([all_libraries, tendl21])
all_libraries = pd.concat([all_libraries, jeff33])
all_libraries = pd.concat([all_libraries, cendl32])

nucs = range_setter(df=endfb8, la=30, ua=210)

all_libraries.index = range(len(all_libraries))


# validation_nuclides = [[39,85], [39,86],[39,87],
# 					   [39,88],[39,90],[39,91],
# 					   [39,92],[39,93],[39,94],
# 					   [39,95]]

# validation_nuclides = []

validation_nuclides = [[40,85], [40,86], [40,87],
					   [40,88], [40,89], [40,93],
					   [40,94], [40,95], [40,96],
					   [40,97], [40,98], [40,99],
					   [40,100], [40,101], [40,102]]

validation_set_size = 30 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(nucs) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides, la=30, ua=215)
X_test, y_test = make_test(validation_nuclides, df=tendl21)

print("Data prep done")

model = xg.XGBRegressor(n_estimators=850,
						learning_rate=0.007,
						max_depth=11,
						subsample=0.1314,
						max_leaves=0,
						seed=42, )

time1= time.time()

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
	nuc = validation_nuclides[i]  # validation nuclide

	endfberg, endfbxs = General_plotter(df=endfb8, nuclides=[nuc])
	endfb_nucs = range_setter(df=endfb8, la=30, ua=210)
	jendlerg, jendlxs = General_plotter(df=jendl5, nuclides=[nuc])
	jendl_nucs = range_setter(df=jendl5, la=30, ua=210)
	cendlerg, cendlxs = General_plotter(df=cendl32, nuclides=[nuc])
	cendl_nucs= range_setter(df=cendl32, la=30, ua=210)
	jefferg, jeffxs = General_plotter(df=jeff33, nuclides=[nuc])
	jeff_nucs = range_setter(df=jeff33, la=30, ua=210)

	plt.plot(erg, pred_xs, label='Predictions', color = 'red', linewidth=2)
	plt.plot(erg, true_xs, label = "TENDL21", color='dimgrey', linewidth=2)
	if nuc in jeff_nucs:
		plt.plot(jefferg, jeffxs, '--', label='JEFF3.3', color='mediumvioletred')
	if nuc in cendl_nucs:
		plt.plot(cendlerg, cendlxs, '--', label='CENDL3.2', color='gold')
	if nuc in jendl_nucs:
		plt.plot(jendlerg, jendlxs, label='JENDL5', color='green')
	if nuc in endfb_nucs:
		plt.plot(endfberg, endfbxs, label='ENDF/B-VIII')
	plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
	plt.legend()
	plt.grid()
	plt.ylabel('XS / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	r2 = r2_score(true_xs, pred_xs)  # R^2 score for this specific nuclide
	print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
	time.sleep(2)

print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}")  # MSE
print(f"R2: {r2_score(y_test, predictions)}")  # Total R^2 for all predictions in this training campaign
print(f'completed in {time.time() - time1} s')