import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
# import shap
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import make_test, make_train, anomaly_remover


# df = pd.read_csv('zeroed_1_xs_fund_feateng.csv') # new interpolated dataset, used for training only
# df_test = pd.read_csv('zeroed_1_xs_fund_feateng.csv') # original dataset, used for validation

# df = pd.read_csv("level_densities_v1_zeroed_1_xs_fund_feateng.csv")
# df_test = pd.read_csv("level_densities_v1_zeroed_1_xs_fund_feateng.csv")  # dataframe as above, but with the new features from the Gilbert-Cameron model

df = pd.read_csv("ENDFBVIII_zeroed_LDP_XS.csv")
df_test = pd.read_csv("ENDFBVIII_zeroed_LDP_XS.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]


# df_test = df_test[df_test.MT == 16] # extract (n,2n) only
df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))



df_test = anomaly_remover(dfa = df_test)
# use once then save dataframe



# def custom_loss(y_pred, y_true):
# 	num = y_pred + 1
# 	den = y_true + 1
#
# 	ln = np.log(y_pred + 1)
#
# 	grad = (num/den) - ln
# 	hess = np.repeat(2, y_true.shape[0])
#
# 	return grad, hess


def range_setter(la, ua):
	nucs = []

	for i, j, in zip(df['A'], df['Z']):  # i is A, j is Z
		if [j, i] in nucs or i > ua or i < la:
			continue
		else:
			nucs.append([j, i])  # format is [Z, A]
	return nucs

al = range_setter(la=30, ua=215)

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

# cat_magic = []
#
# magic_numbers = [2, 8, 20, 28, 50, 82, 126]
#
#
# cat_magic_double_original = []
# cat_magic_neutron_original = []
# cat_magic_proton_original = []
#
# for z, n in zip(df_test['Z'], df_test['A']):
# 	if z in magic_numbers and n in magic_numbers:
# 		cat_magic_proton_original.append(1)
# 		cat_magic_double_original.append(1)
# 		cat_magic_neutron_original.append(1)
# 	elif z in magic_numbers and n not in magic_numbers:
# 		cat_magic_proton_original.append(1)
# 		cat_magic_neutron_original.append(0)
# 		cat_magic_double_original.append(0)
# 	elif z not in magic_numbers and n in magic_numbers:
# 		cat_magic_neutron_original.append(1)
# 		cat_magic_double_original.append(0)
# 		cat_magic_proton_original.append(0)
# 	else:
# 		cat_magic_proton_original.append(0)
# 		cat_magic_double_original.append(0)
# 		cat_magic_neutron_original.append(0)
#
# df_test.insert(78, 'cat_magic_proton', cat_magic_proton_original)
# df_test.insert(79, 'cat_magic_neutron', cat_magic_neutron_original)
# df_test.insert(80, 'cat_magic_double', cat_magic_double_original)
#
# df_test['cat_magic_proton'].astype('category')
# df_test['cat_magic_neutron'].astype('category')
# df_test['cat_magic_double'].astype("category")
#
#
# cat_magic_proton = []
# cat_magic_neutron = []
# cat_magic_double = []
#
# for z, n in zip(df['Z'], df['N']):
# 	if z in magic_numbers and n in magic_numbers:
# 		cat_magic_double.append(1)
# 		cat_magic_neutron.append(1)
# 		cat_magic_proton.append(1)
# 	elif z in magic_numbers and n not in magic_numbers:
# 		cat_magic_proton.append(1)
# 		cat_magic_neutron.append(0)
# 		cat_magic_double.append(0)
# 	elif z not in magic_numbers and n in magic_numbers:
# 		cat_magic_neutron.append(1)
# 		cat_magic_proton.append(0)
# 		cat_magic_double.append(0)
# 	else:
# 		cat_magic_proton.append(0)
# 		cat_magic_double.append(0)
# 		cat_magic_neutron.append(0)
#
#
# df.insert(78, 'cat_magic_proton', cat_magic_proton)
# df.insert(79, 'cat_magic_neutron', cat_magic_neutron)
# df.insert(80, 'cat_magic_double', cat_magic_double)
#
# df['cat_magic_proton'].astype('category')
# df['cat_magic_neutron'].astype('category')
# df['cat_magic_double'].astype("category")

# validation_nuclides = [[82,208]] # list of nuclides used for validation
# validation_nuclides = [[20,40], [28,58], [24,50]]
validation_nuclides = []
validation_set_size = 20 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")



if __name__ == "__main__":

	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=50, ua=215) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=df_test)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Data prep done")


	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.015,
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

	print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
	print(f"R2: {r2_score(y_test, predictions)}") # Total R^2 for all predictions in this training campaign
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
										 'rms_r',
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
										 'Ntop',
										 'Utop',
										 'ainf',
										 ]

	# Gain-based feature importance plot
	plt.figure(figsize=(10, 12))
	xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
	plt.show()

	# Splits-based feature importance plot
	# plt.figure(figsize=(10, 12))
	# plt.title("Splits-based FI")
	# xg.plot_importance(model, ax=plt.gca(), max_num_features=60)
	# plt.show()

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
	# 										   'rms_r',
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
	# 										   'Ntop',
	# 										   'Utop',
	# 										   'ainf',
	# 										   ]) # SHAP feature importance analysis
	# shap_values = explainer(X_test)

	# name features for FIA


	# shap.plots.bar(shap_values, max_display = 70) # display SHAP results
	# shap.plots.waterfall(shap_values[0], max_display=70)
