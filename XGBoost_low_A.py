import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
import shap
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import log_make_test_low, log_make_train_low, anomaly_remover, range_setter

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]


# df_test = df_test[df_test.MT == 16] # extract (n,2n) only
df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))

log_red_var = 0.00001


df_test = anomaly_remover(dfa = df_test)

al = range_setter(la=0, ua=50, df=df)

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


validation_nuclides = []
validation_set_size = 5 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")



if __name__ == "__main__":

	X_train, y_train = log_make_train_low(df=df, validation_nuclides=validation_nuclides,
										  la=0, ua=50, log_reduction_variable=log_red_var) # make training matrix

	X_test, y_test = log_make_test_low(validation_nuclides, df=df_test,log_reduction_variable=log_red_var)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Data prep done")


	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.005,
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
	for nuclide in validation_nuclides:
		dummy_test_XS = []
		dummy_test_E = []
		dummy_predictions = []
		for i, row in enumerate(X_test):
			if [row[0], row[1]] == nuclide:
				dummy_test_XS.append(np.exp(y_test[i]) - log_red_var)
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(np.exp(predictions[i]) - log_red_var)

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i] # validation nuclide
		plt.plot(erg, pred_xs, label='predictions')
		plt.plot(erg, true_xs, label='data')
		plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
		plt.legend()
		plt.grid()
		plt.ylabel('XS / b')
		plt.xlabel('Energy / MeV')
		plt.show()

		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
		time.sleep(0.7)

	exp_true_xs = [np.exp(y) -log_red_var for y in y_test]
	exp_pred_xs = [np.exp(xs)- log_red_var for xs in predictions]

	print(f"MSE: {mean_squared_error(exp_true_xs, exp_pred_xs, squared=False)}")  # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs):0.5f}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')

	model.get_booster().feature_names = ['Z', 'A', 'S2n', 'S2p', 'E', 'Sp', 'Sn',
										 'BEA',
										 # 'P',
										 'Snc', 'g-def',
										 'N',
										 'b-def',
										 'Sn da',
										 'Sp d',
										 'S2n d',
										 'Radius',
										 'n_g_erg',
										 'n_c_erg',
										 # 'xsmax',
										 'n_rms_r',
										 # 'oct_def',
										 'D_c',
										 'BEA_d',
										 'BEA_c',
										 # 'Pair_d',
										 # 'Par_d',
										 'S2n_c',
										 'S2p_c',
										 'ME',
										 # 'Z_even',
										 # 'A_even',
										 # 'N_even',
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
										 # 'Sn_c',
										 'Shell_c',
										 # 'S2p-d',
										 # 'Shell-d',
										 'Spin-c',
										 'Rad-c',
										 'Def-c',
										 'ME-c',
										 'BEA-A-c',
										 'Decay-d',
										 'ME-d',
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
										 'Ulow',
										 # 'Ntop',
										 'Utop',
										 'ainf',
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

	# explainer = shap.Explainer(model.predict, X_train,
	# 						   feature_names= ['Z', 'A', 'S2n', 'S2p', 'E', 'Sp', 'Sn',
	# 										   'BEA',
	# 										   # 'P',
	# 										   'Snc', 'g-def', 'N',
	# 										   'b-def',
	# 										   'Sn_da',
	# 										   'Sp_d',
	# 										   'S2n_d',
	# 										   'Radius',
	# 										   'n_g_erg',
	# 										   'n_c_erg',
	# 										   # 'xsmax',
	# 										   'n_rms_r',
	# 										   # 'oct_def',
	# 										   'D_c',
	# 										   'BEA_d',
	# 										   'BEA_c',
	# 										   # 'Pair_d',
	# 										   # 'Par_d',
	# 										   'S2n_c',
	# 										   'S2p_c',
	# 										   'ME',
	# 										   # 'Z_even',
	# 										   # 'A_even',
	# 										   # 'N_even',
	# 										   'Shell',
	# 										   'Par',
	# 										   'Spin',
	# 										   'Decay',
	# 										   'Deform',
	# 										   'p_g_e',
	# 										   'p_c_e',
	# 										   'p_rms_r',
	# 										   # 'rms_r',
	# 										   'Sp_c',
	# 										   # 'S_n_c',
	# 										   'Shell_c',
	# 										   # 'S2p-d',
	# 										   # 'Shell-d',
	# 										   'Spin-c',
	# 										   'Rad-c',
	# 										   'Def-c',
	# 										   'ME-c',
	# 										   'BEA-A-c',
	# 										   'Decay-d',
	# 										   'ME-d',
	# 										   # 'Rad-d',
	# 										   # 'Pair-c',
	# 										   # 'Par-c',
	# 										   'BEA-A-d',
	# 										   # 'Spin-d',
	# 										   'Def-d',
	# 										   # 'mag_p',
	# 										   # 'mag_n',
	# 										   # 'mag_d',
	# 										   'Nlow',
	# 										   'Ulow',
	# 										   # 'Ntop',
	# 										   'Utop',
	# 										   'ainf',
	# 										   ]) # SHAP feature importance analysis
	# shap_values = explainer(X_test)
	#
	# name features for FIA
	#
	#
	# shap.plots.bar(shap_values, max_display = 70) # display SHAP results
	# shap.plots.waterfall(shap_values[0], max_display=70)
