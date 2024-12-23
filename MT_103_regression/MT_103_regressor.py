import pandas as pd
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
from MT_103_functions import make_train, make_test, range_setter, r2_standardiser, Generalplotter103
import matplotlib.pyplot as plt
import xgboost
import periodictable
import random
import time
import shap
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('ENDFBVIII_MT_103_all_features.csv')
CENDL_32 = pd.read_csv('CENDL-3.2_MT_103_all_features.csv')
CENDL_nuclides = range_setter(df=CENDL_32)

JEFF_33 = pd.read_csv('JEFF-3.3_MT_103_all_features.csv')
JEFF_nuclides = range_setter(df=JEFF_33)

JENDL_5 = pd.read_csv('JENDL-5_MT_103_all_features.csv')
JENDL_nuclides = range_setter(df=JENDL_5)

TENDL_2021 = pd.read_csv('TENDL-2021_MT_103_all_features.csv')
TENDL_nuclides = range_setter(df=TENDL_2021)

ENDFB_nuclides = range_setter(df=df, la=0, ua=208)
print("Data loaded...")

validation_nuclides = [[53,129]]
validation_set_size = 20

energy_row = 4

min_energy = 0
max_energy = 20

mode = input("Mode (thr/1v/both): ")

while len(validation_nuclides) < validation_set_size:
	nuclide_choice = random.choice(ENDFB_nuclides) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
	qchoice = df[(df['Z'] == nuclide_choice[0]) & (df['A'] == nuclide_choice[1])]['Q'].values[0]
	if nuclide_choice not in validation_nuclides and qchoice < 0 and mode == "thr":
		validation_nuclides.append(nuclide_choice)
	elif nuclide_choice not in validation_nuclides:
		validation_nuclides.append(nuclide_choice)

print("Test nuclides selected...")

X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=208, minerg=min_energy, maxerg=max_energy,
							  mode='threshold')
X_test, y_test = make_test(validation_nuclides, df=df, minerg=min_energy, maxerg=max_energy)

model = xgboost.XGBRegressor(n_estimators = 1100,
							 learning_rate = 0.007,
							 max_depth = 7,
							 subsample = 0.888,
							 reg_lambda = 4
							 )

model.fit(X_train, y_train,verbose=True, eval_set=[(X_test, y_test)])
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
			dummy_test_E.append(row[energy_row])  # Energy values are in 5th row
			dummy_predictions.append(predictions[i])

	XS_plotmatrix.append(dummy_test_XS)
	E_plotmatrix.append(dummy_test_E)
	P_plotmatrix.append(dummy_predictions)

print("Plotting matrices formed")

for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
	all_preds = []
	all_libs = []
	current_nuclide = validation_nuclides[i]

	q = df[(df['Z'] == current_nuclide[0]) & (df['A'] == current_nuclide[1])]['Q'].values[0]

	jendlerg, jendlxs = Generalplotter103(dataframe=JENDL_5, nuclide=current_nuclide)
	cendlerg, cendlxs = Generalplotter103(dataframe=CENDL_32, nuclide=current_nuclide)
	jefferg, jeffxs = Generalplotter103(dataframe=JEFF_33, nuclide=current_nuclide)
	tendlerg, tendlxs = Generalplotter103(dataframe=TENDL_2021, nuclide=current_nuclide)

	nuc = validation_nuclides[i]  # validation nuclide
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
	plt.plot(tendlerg, tendlxs, label="TENDL-2021", color='dimgrey', linewidth=2)
	if nuc in JENDL_nuclides:
		plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
	if nuc in CENDL_nuclides:
		plt.plot(cendlerg, cendlxs, '--', label='CENDL-3.2', color='gold')

	plt.title(f"$\sigma_{{n,p}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]:0.0f}")
	plt.legend()
	plt.grid()
	plt.ylabel('$\sigma_{n,p}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	time.sleep(1)
	print(f'{periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]:0.0f}')
	print(f"Q: {q} eV, +ve no threshold, -ve threshold")
	print(f'Predictions - ENDF/B R2: {r2_score(pred_xs, true_xs):0.5f}')

	if current_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL_32, minerg=min_energy, maxerg=max_energy)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl, library_xs=cendl_xs)

		pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
		for x, y in zip(pred_cendl_gated, truncated_cendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if current_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL_5, minerg=min_energy, maxerg=max_energy)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl, library_xs=jendl_xs)

		for x, y in zip(pred_jendl_gated, truncated_jendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if current_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF_33, minerg=min_energy, maxerg=max_energy)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if current_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL_2021, minerg=min_energy, maxerg=max_energy)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl, library_xs=tendl_xs)
		for x, y in zip(pred_tendl_gated, truncated_tendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

	print()
run = input('Run shap? (y): ')
if run == 'y':

	model.get_booster().feature_names = ['Z',
										 'A',
										 'S2n',
										 'S2p',
										 'E',
										 'Sp',
										 'Sn',
										 'BEA',
										 'P',
										 'Snc',
										 'g-def',
										 'N',
										 'b-def',
										 'Sn da',
										 'Sp d',
										 'S2n d',
										 'Radius',
										 'n_g_erg',
										 'n_c_erg',
										 'n_rms_r',
										 # 'oct_def',
										 'D_c',
										 'BEA_d',
										 'BEA_c',
										 'Pair_d',
										 'Par_d',
										 'S2n_c',
										 'S2p_c',
										 'ME',
										 'Z_even',
										 'A_even',
										 'N_even',
										 'Shell',
										 'Parity',
										 'Spin',
										 'Decay',
										 'Deform',
										 'p_g_e',
										 'p_c_e',
										 'p_rms_r',
										 # 'rms_r',
										 'Sp_c',
										 'Sn_c',
										 'Shell_c',
										 'S2p-d',
										 'Shell-d',
										 'Spin-c',
										 'Rad-c',
										 'Def-c',
										 'ME-c',
										 'BEA-A-c',
										 'Decay-d',
										 'ME-d',
										 'Rad-d',
										 'Pair-c',
										 'Par-c',
										 'BEA-A-d',
										 'Spin-d',
										 'Def-d',
										 # 'mag_p',
										 # 'mag-n',
										 # 'mag-d',
										 # 'Nlow',
										 # 'Ulow',
										 # 'Ntop',
										 # 'Utop',
										 # 'ainf',
										 'Asym',
										 'Asym_c',
										 'Asym_d',
										 # 'AM',
										 'Q'
										 ]
	plt.figure(figsize=(10, 12))
	xgboost.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
	plt.show()

	explainer = shap.Explainer(model.predict, X_train,
							   feature_names= ['Z',
										 'A',
										 'S2n',
										 'S2p',
										 'E',
										 'Sp',
										 'Sn',
										 'BEA',
										 'P',
										 'Snc',
										 'g-def',
										 'N',
										 'b-def',
										 'Sn da',
										 'Sp d',
										 'S2n d',
										 'Radius',
										 'n_g_erg',
										 'n_c_erg',
										 'n_rms_r',
										 # 'oct_def',
										 'D_c',
										 'BEA_d',
										 'BEA_c',
										 'Pair_d',
										 'Par_d',
										 'S2n_c',
										 'S2p_c',
										 'ME',
										 'Z_even',
										 'A_even',
										 'N_even',
										 'Shell',
										 'Parity',
										 'Spin',
										 'Decay',
										 'Deform',
										 'p_g_e',
										 'p_c_e',
										 'p_rms_r',
										 # 'rms_r',
										 'Sp_c',
										 'Sn_c',
										 'Shell_c',
										 'S2p-d',
										 'Shell-d',
										 'Spin-c',
										 'Rad-c',
										 'Def-c',
										 'ME-c',
										 'BEA-A-c',
										 'Decay-d',
										 'ME-d',
										 'Rad-d',
										 'Pair-c',
										 'Par-c',
										 'BEA-A-d',
										 'Spin-d',
										 'Def-d',
										 # 'mag_p',
										 # 'mag-n',
										 # 'mag-d',
										 # 'Nlow',
										 # 'Ulow',
										 # 'Ntop',
										 # 'Utop',
										 # 'ainf',
										 'Asym',
										 'Asym_c',
										 'Asym_d',
										 'Q',
										 # 'AM'
										 ]) # SHAP feature importance analysis
	shap_values = explainer(X_test)



	shap.plots.bar(shap_values, max_display = 70) # display SHAP results