import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable
import scipy
from matrix_functions import General_plotter, range_setter, mtmake_train, mtmake_test, dsigma_dE, r2_standardiser
import random
# import shap
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error
import time
import shap

TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=210)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=210)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=210)



df = pd.read_csv('ENDFBVIII_MT16_MT103_MT107_fund_features.csv')


df.index = range(len(df))
# df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=0, ua=100, df=df)

validation_nuclides = [[30,64]]
validation_set_size = 15

# random.seed(a=10)

while len(validation_nuclides) < validation_set_size: # up to 25 nuclides
	choice = random.choice(al) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


X_train, y_train = mtmake_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=100, exclusions=[]) # create training matrix
X_test, y_test = mtmake_test(validation_nuclides, df=df,) # create test matrix using validation nuclides
print("Data prep done")

model = xg.XGBRegressor(n_estimators=950, # define regressor
						learning_rate=0.008,
						max_depth=8,
						subsample=0.25,
						max_leaves=0,
						seed=42,
						reg_lambda = 0,
						)

model.fit(X_train, y_train)
print("Training complete")
predictions = model.predict(X_test)  # XS predictions
predictions_ReLU = []

for pred in predictions: # prediction gate
	if pred >= 0.003:
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
	plt.plot(tendlerg, tendlxs, label="TENDL21", color='dimgrey', linewidth=2)
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF3.3', color='mediumvioletred')
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

	mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)
	print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
	time.sleep(0.8)

	pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
	pred_endfb_gated, truncated_endfb, pred_endfb_r2 = r2_standardiser(predicted_xs=pred_xs, library_xs=true_xs)
	for x, y in zip(pred_endfb_gated, truncated_endfb):
		all_libs.append(y)
		all_preds.append(x)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")


# FIA
# model.get_booster().feature_names = ['Z',
# 									 'A',
# 									 'S2n',
# 									 'S2p',
# 									 'E',
# 									 'Sp',
# 									 'Sn',
# 									 'BEA',
# 									 # 'P',
# 									 'Snc',
# 									 'g-def',
# 									 'N',
# 									 'b-def',
# 									 'Sn da',
# 									 # 'Sp d',
# 									 'S2n d',
# 									 'Radius',
# 									 'n_g_erg',
# 									 'n_c_erg',
# 									 'n_rms_r',
# 									 # 'oct_def',
# 									 # 'D_c',
# 									 'BEA_d',
# 									 'BEA_c',
# 									 # 'Pair_d',
# 									 # 'Par_d',
# 									 # 'S2n_c',
# 									 'S2p_c',
# 									 'ME',
# 									 # 'Z_even',
# 									 # 'A_even',
# 									 # 'N_even',
# 									 'Shell',
# 									 # 'Parity',
# 									 # 'Spin',
# 									 'Decay',
# 									 # 'Deform',
# 									 'p_g_e',
# 									 'p_c_e',
# 									 # 'p_rms_r',
# 									 # 'rms_r',
# 									 # 'Sp_c',
# 									 # 'Sn_c',
# 									 'Shell_c',
# 									 # 'S2p-d',
# 									 # 'Shell-d',
# 									 'Spin-c',
# 									 # 'Rad-c',
# 									 'Def-c',
# 									 # 'ME-c',
# 									 'BEA-A-c',
# 									 'Decay-d',
# 									 # 'ME-d',
# 									 # 'Rad-d',
# 									 # 'Pair-c',
# 									 # 'Par-c',
# 									 'BEA-A-d',
# 									 # 'Spin-d',
# 									 'Def-d',
# 									 # 'mag_p',
# 									 # 'mag-n',
# 									 # 'mag-d',
# 									 'Nlow',
# 									 # 'Ulow',
# 									 # 'Ntop',
# 									 'Utop',
# 									 # 'ainf',
# 									 'Asym',
# 									 # 'Asym_c',
# 									 'Asym_d',
# 									 # 'AM',
# 									 'MT103_XS',
# 									 'MT107_XS',
# 									 # 'MT91_XS'
# 									 ]
#
# # Gain-based feature importance plot
# plt.figure(figsize=(10, 12))
# xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
# plt.show()
#
# # Splits-based feature importance plot
# plt.figure(figsize=(10, 12))
# plt.title("Splits-based FI")
# xg.plot_importance(model, ax=plt.gca(), max_num_features=60)
# plt.show()
#
# explainer = shap.Explainer(model.predict, X_train,
# 						   feature_names= ['Z',
# 									 'A',
# 									 'S2n',
# 									 'S2p',
# 									 'E',
# 									 'Sp',
# 									 'Sn',
# 									 'BEA',
# 									 # 'P',
# 									 'Snc',
# 									 'g-def',
# 									 'N',
# 									 'b-def',
# 									 'Sn da',
# 									 # 'Sp d',
# 									 'S2n d',
# 									 'Radius',
# 									 'n_g_erg',
# 									 'n_c_erg',
# 									 'n_rms_r',
# 									 # 'oct_def',
# 									 # 'D_c',
# 									 'BEA_d',
# 									 'BEA_c',
# 									 # 'Pair_d',
# 									 # 'Par_d',
# 									 # 'S2n_c',
# 									 'S2p_c',
# 									 'ME',
# 									 # 'Z_even',
# 									 # 'A_even',
# 									 # 'N_even',
# 									 'Shell',
# 									 # 'Parity',
# 									 # 'Spin',
# 									 'Decay',
# 									 # 'Deform',
# 									 'p_g_e',
# 									 'p_c_e',
# 									 # 'p_rms_r',
# 									 # 'rms_r',
# 									 # 'Sp_c',
# 									 # 'Sn_c',
# 									 'Shell_c',
# 									 # 'S2p-d',
# 									 # 'Shell-d',
# 									 'Spin-c',
# 									 # 'Rad-c',
# 									 'Def-c',
# 									 # 'ME-c',
# 									 'BEA-A-c',
# 									 'Decay-d',
# 									 # 'ME-d',
# 									 # 'Rad-d',
# 									 # 'Pair-c',
# 									 # 'Par-c',
# 									 'BEA-A-d',
# 									 # 'Spin-d',
# 									 'Def-d',
# 									 # 'mag_p',
# 									 # 'mag-n',
# 									 # 'mag-d',
# 									 'Nlow',
# 									 # 'Ulow',
# 									 # 'Ntop',
# 									 'Utop',
# 									 # 'ainf',
# 									 'Asym',
# 									 # 'Asym_c',
# 									 'Asym_d',
# 									 # 'AM',
# 									 'MT103XS',
# 									 'MT107XS',
# 									 # 'MT_91XS',
# 									 ]) # SHAP feature importance analysis
# shap_values = explainer(X_test)
#
#
#
# shap.plots.bar(shap_values, max_display = 70) # display SHAP results
# shap.plots.waterfall(shap_values[0], max_display=70)
