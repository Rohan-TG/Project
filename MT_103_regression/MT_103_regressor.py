import pandas as pd
from MT_103_functions import make_train, make_test, range_setter
import matplotlib.pyplot as plt
import xgboost
import periodictable
import random
import time

df = pd.read_csv('ENDFBVIII_MT103_fund_features_only.csv')

ENDFB_nuclides = range_setter(df=df, la=0, ua=260)
print("Data loaded")

validation_nuclides = []
validation_set_size = 10

while len(validation_nuclides) < validation_set_size:
	nuclide_choice = random.choice(ENDFB_nuclides) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
	if nuclide_choice not in validation_nuclides:
		validation_nuclides.append(nuclide_choice)
print("Test nuclide selection complete")

X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=260)
X_test, y_test = make_test(validation_nuclides, df=df)

model = xgboost.XGBRegressor(n_estimators = 900,
							 learning_rate = 0.008,
							 max_depth = 8,
							 subsample = 0.2
							 )

model.fit(X_train, y_train)
print("Training complete")
predictions = model.predict(X_test)

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

print("Plotting matrices formed")

for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
	all_preds = []
	all_libs = []
	current_nuclide = validation_nuclides[i]

	nuc = validation_nuclides[i]  # validation nuclide
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)

	plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
	plt.legend()
	plt.grid()
	plt.ylabel('$\sigma_{n,2n}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	time.sleep(1)