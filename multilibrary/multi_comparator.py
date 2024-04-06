from matrix_functions import make_test, make_train, range_setter, General_plotter, r2_standardiser
import pandas as pd
import xgboost as xg
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import time
import shap
import periodictable



ENDFB = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
TENDL = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JEFF = pd.read_csv('JEFF33_all_features.csv')
CENDL = pd.read_csv('CENDL32_all_features.csv')

ENDFB_nuclides = range_setter(df=ENDFB, la=30, ua=210)
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)

all_libraries = pd.read_csv('MT16_all_libraries_mk1.csv')
all_libraries.index = range(len(all_libraries))

exc = [[22, 47], [65, 159], [66, 157], [38, 90], [61, 150],
	   [74, 185], [50, 125], [50, 124], [60, 149], [39, 90],
	   [64, 160], [38, 87], [39, 91], [63, 152], [52, 125],
	   [19, 40], [56, 139], [52, 126], [71, 175], [34, 79],
	   [70, 175], [50, 117], [23, 49], [63, 156], [57, 140],
	   [52, 128], [59, 142], [50, 118], [50, 123], [65, 161],
	   [52, 124], [38, 85], [51, 122], [19, 41], [54, 135],
	   [32, 75], [81, 205], [71, 176], [72, 175], [50, 122],
	   [51, 125], [53, 133], [34, 82], [41, 95], [46, 109],
	   [84, 209], [56, 140], [64, 159], [68, 167], [16, 35],
	   [18,38], [44,99], [50,126]] # 10 sigma with handpicked additions



validation_nuclides = [[71,175], [30,64]]
validation_set_size = 20

while len(validation_nuclides) < validation_set_size: # up to 25 nuclides
	choice = random.choice(ENDFB_nuclides) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides, la=30, ua=210,
							  exclusions=exc) # create training matrix
X_test, y_test = make_test(validation_nuclides, df=ENDFB,) # create test matrix using validation nuclides
print("Data preparation complete. Training...")

model = xg.XGBRegressor(n_estimators=1200, # define regressor
						learning_rate=0.0035,
						max_depth=11,
						subsample=0.1556,
						max_leaves=0,
						seed=42, )
time1 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {(time.time() - time1)} seconds")
print("Training complete. Processing/Plotting...")
predictions = model.predict(X_test)  # XS predictions
predictions_ReLU = []

for pred in predictions: # prediction gate
	if pred >= 0.004:
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
	all_preds = []
	all_libs = []
	current_nuclide = validation_nuclides[i]

	jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
	cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
	jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
	tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])

	nuc = validation_nuclides[i]  # validation nuclide
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
	plt.plot(tendlerg, tendlxs, label="TENDL-2021", color='dimgrey', linewidth=2)
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
	if nuc in JENDL_nuclides:
		plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
	if nuc in CENDL_nuclides:
		plt.plot(cendlerg, cendlxs, '--', label='CENDL-3.2', color='gold')
	plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
	plt.legend()
	plt.grid()
	plt.ylabel('$\sigma_{n,2n}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)
	print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
	time.sleep(0.8)

	pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
	pred_endfb_gated, truncated_endfb, pred_endfb_r2 = r2_standardiser(raw_predictions=pred_xs, library_xs=true_xs)

	endfbmape = mean_absolute_percentage_error(truncated_endfb, pred_endfb_gated)
	for x, y in zip(pred_endfb_gated, truncated_endfb):
		all_libs.append(y)
		all_preds.append(x)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MAPE: {endfbmape:0.4f}")

	# Find r2 w.r.t. other libraries
	if current_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl, library_xs=cendl_xs)

		pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
		for x, y in zip(pred_cendl_gated, truncated_cendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if current_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl, library_xs=jendl_xs)

		for x, y in zip(pred_jendl_gated, truncated_jendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if current_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if current_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl, library_xs=tendl_xs)
		for x, y in zip(pred_tendl_gated, truncated_tendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

	print(f"Consensus R2: {r2_score(all_libs, all_preds):0.5f}")

print(f'Completed in {time.time() - time1} s.')

run = input('Run FIA? (y): ')
if run == 'y':
	# FIA
	model.get_booster().feature_names = ['Z',
										 'A',
										 'S2n',
										 'S2p',
										 'E',
										 'Sp',
										 'Sn',
										 'BEA',
										 # 'P',
										 'Snc',
										 'g-def',
										 'N',
										 'b-def',
										 'Sn da',
										 # 'Sp d',
										 'S2n d',
										 'Radius',
										 'n_g_erg',
										 'n_c_erg',
										 'n_rms_r',
										 # 'oct_def',
										 # 'D_c',
										 'BEA_d',
										 'BEA_c',
										 # 'Pair_d',
										 # 'Par_d',
										 # 'S2n_c',
										 'S2p_c',
										 'ME',
										 # 'Z_even',
										 # 'A_even',
										 # 'N_even',
										 'Shell',
										 # 'Parity',
										 # 'Spin',
										 'Decay',
										 # 'Deform',
										 'p_g_e',
										 'p_c_e',
										 # 'p_rms_r',
										 # 'rms_r',
										 # 'Sp_c',
										 # 'Sn_c',
										 'Shell_c',
										 # 'S2p-d',
										 # 'Shell-d',
										 'Spin-c',
										 # 'Rad-c',
										 'Def-c',
										 # 'ME-c',
										 'BEA-A-c',
										 'Decay-d',
										 # 'ME-d',
										 # 'Rad-d',
										 # 'Pair-c',
										 # 'Par-c',
										 'BEA-A-d',
										 # 'Spin-d',
										 'Def-d',
										 # 'mag_p',
										 # 'mag-n',
										 # 'mag-d',
										 'Nlow',
										 # 'Ulow',
										 # 'Ntop',
										 'Utop',
										 # 'ainf',
										 'Asym',
										 # 'Asym_c',
										 'Asym_d',
										 # 'AM'
										 ]

	# Gain-based feature importance plot
	plt.figure(figsize=(10, 12))
	xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
	plt.show()

	# Splits-based feature importance plot
	plt.figure(figsize=(10, 12))
	plt.title("Splits-based FI")
	xg.plot_importance(model, ax=plt.gca(), max_num_features=60)
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
										 # 'P',
										 'Snc',
										 'g-def',
										 'N',
										 'b-def',
										 'Sn da',
										 # 'Sp d',
										 'S2n d',
										 'Radius',
										 'n_g_erg',
										 'n_c_erg',
										 'n_rms_r',
										 # 'oct_def',
										 # 'D_c',
										 'BEA_d',
										 'BEA_c',
										 # 'Pair_d',
										 # 'Par_d',
										 # 'S2n_c',
										 'S2p_c',
										 'ME',
										 # 'Z_even',
										 # 'A_even',
										 # 'N_even',
										 'Shell',
										 # 'Parity',
										 # 'Spin',
										 'Decay',
										 # 'Deform',
										 'p_g_e',
										 'p_c_e',
										 # 'p_rms_r',
										 # 'rms_r',
										 # 'Sp_c',
										 # 'Sn_c',
										 'Shell_c',
										 # 'S2p-d',
										 # 'Shell-d',
										 'Spin-c',
										 # 'Rad-c',
										 'Def-c',
										 # 'ME-c',
										 'BEA-A-c',
										 'Decay-d',
										 # 'ME-d',
										 # 'Rad-d',
										 # 'Pair-c',
										 # 'Par-c',
										 'BEA-A-d',
										 # 'Spin-d',
										 'Def-d',
										 # 'mag_p',
										 # 'mag-n',
										 # 'mag-d',
										 'Nlow',
										 # 'Ulow',
										 # 'Ntop',
										 'Utop',
										 # 'ainf',
										 'Asym',
										 # 'Asym_c',
										 'Asym_d',
										 # 'AM'
										 ]) # SHAP feature importance analysis
	shap_values = explainer(X_test)
	#
	#
	#
	shap.plots.bar(shap_values, max_display = 70) # display SHAP results
else:
	print('Completed.')