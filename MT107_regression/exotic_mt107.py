import time
import pandas as pd
import xgboost as xg
from matrix_functions import range_setter, r2_standardiser
import random
from functions107 import maketest107, maketrain107, Generalplotter107
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import periodictable










df = pd.read_csv('ENDFBVIII_MT_107_all_features.csv')
df = df[(df['Z'] != 6) & (df['A'] != 12)]
df.index = range(len(df))
CENDL_32 = pd.read_csv('CENDL-3.2_MT_107_all_features.csv')
CENDL_nuclides = range_setter(df=CENDL_32, la=0, ua=208)
#
JEFF_33 = pd.read_csv('JEFF-3.3_MT_107_all_features.csv')
JEFF_nuclides = range_setter(df=JEFF_33, la=0, ua=208)
#
JENDL_5 = pd.read_csv('JENDL-5_MT_107_all_features.csv')
JENDL_nuclides = range_setter(df=JENDL_5, la=0, ua=208)

#
TENDL_2021 = pd.read_csv('TENDL-2021_MT_107_all_features.csv')
TENDL_nuclides = range_setter(df=TENDL_2021, la=0, ua=208)

ENDFB_nuclides = range_setter(df=df, la=0, ua=208)
print("Data loaded...")

validation_nuclides = []
validation_set_size = 20

minenergy = 0.1
maxenergy = 20









def diff(target):
	match_element = []
	zl = [n[0] for n in ENDFB_nuclides]
	if target[0] in zl:
		for nuclide in ENDFB_nuclides:
			if nuclide[0] == target[0]:
				match_element.append(nuclide)
		nuc_differences = []
		for match_nuclide in match_element:
			difference = match_nuclide[1] - target[1]
			nuc_differences.append(difference)
		if len(nuc_differences) > 0:
			if nuc_differences[0] > 0:
				min_difference = min(nuc_differences)
			elif nuc_differences[0] < 0:
				min_difference = max(nuc_differences)
		return (-1 * min_difference)








#
exotic_nuclides = []
for n in TENDL_nuclides:
	if n not in ENDFB_nuclides:
		exotic_nuclides.append(n)

while len(validation_nuclides) < validation_set_size:
	nuclide_choice = random.choice(exotic_nuclides) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
	choicediff = diff(nuclide_choice)
	if nuclide_choice not in validation_nuclides:
		try:
			absval = abs(choicediff)
			if absval < 4:
				validation_nuclides.append(nuclide_choice)
		except:
			continue
		# validation_nuclides.append(nuclide_choice)



X_train, y_train = maketrain107(df=df, validation_nuclides=validation_nuclides,
								la=0, ua=208, maxerg=maxenergy, minerg=minenergy)
X_test, y_test = maketest107(validation_nuclides, df=TENDL_2021)

model = xg.XGBRegressor(n_estimators = 900,
							 learning_rate = 0.01,
							 max_depth = 7,
							 subsample = 0.2,
							 reg_lambda = 2
							 )

model.fit(X_train, y_train,verbose=True, eval_set=[(X_test, y_test)])
print("Training complete")


predictions_r = []
for n in validation_nuclides:

	temp_x, temp_y = maketest107(nuclides=[n], df=TENDL_2021)
	initial_predictions = model.predict(temp_x)

	for p in initial_predictions:
		if p >= (0.02 * max(initial_predictions)):
			predictions_r.append(p)
		else:
			predictions_r.append(0.0)

predictions = predictions_r
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
	mass_difference = diff(current_nuclide)

	q =TENDL_2021[(TENDL_2021['Z'] == current_nuclide[0]) & (TENDL_2021['A'] == current_nuclide[1])]['Q'].values[0]

	jendlerg, jendlxs = Generalplotter107(dataframe=JENDL_5, nuclide=current_nuclide)
	cendlerg, cendlxs = Generalplotter107(dataframe=CENDL_32, nuclide=current_nuclide)
	jefferg, jeffxs = Generalplotter107(dataframe=JEFF_33, nuclide=current_nuclide)
	tendlerg, tendlxs = Generalplotter107(dataframe=TENDL_2021, nuclide=current_nuclide)

	nuc = validation_nuclides[i]  # validation nuclide
	plt.figure()
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	# plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
	plt.plot(tendlerg, tendlxs, label="TENDL-2021", color='dimgrey', linewidth=2)
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
	if nuc in JENDL_nuclides:
		plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
	if nuc in CENDL_nuclides:
		plt.plot(cendlerg, cendlxs, '--', label='CENDL-3.2', color='gold')

	plt.title(fr"$\sigma_{{n,\alpha}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]:0.0f}")
	plt.legend()
	plt.grid()
	plt.ylabel(r'$\sigma_{n,\alpha}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()
	time.sleep(1.5)
	print(f'{periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}')
	print(f"Q: {q} eV +ve no threshold, -ve threshold")

	endfbpredtruncated, endfblibtruncated, endfbr2 = r2_standardiser(library_xs=true_xs, predicted_xs=pred_xs)
	for x, y in zip(endfbpredtruncated, endfblibtruncated):
		all_libs.append(y)
		all_preds.append(x)
	print(f"Predictions - ENDF/B-VIII: {endfbr2:0.5f}")

	if current_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = maketest107(nuclides=[current_nuclide], df=CENDL_32)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(predicted_xs=pred_cendl, library_xs=cendl_xs)

		pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
		for x, y in zip(pred_cendl_gated, truncated_cendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if current_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = maketest107(nuclides=[current_nuclide], df=JENDL_5)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(predicted_xs=pred_jendl, library_xs=jendl_xs)

		for x, y in zip(pred_jendl_gated, truncated_jendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if current_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = maketest107(nuclides=[current_nuclide], df=JEFF_33)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(predicted_xs=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if current_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = maketest107(nuclides=[current_nuclide], df=TENDL_2021)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(predicted_xs=pred_tendl, library_xs=tendl_xs)
		for x, y in zip(pred_tendl_gated, truncated_tendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")
	print(f'Mass difference: {mass_difference}')
	print()

