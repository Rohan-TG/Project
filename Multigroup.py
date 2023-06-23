from __future__ import annotations
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

df = pd.read_csv('1_xs_fund_feateng.csv') # new interpolated dataset, used for training only
df_test = pd.read_csv('1_xs_fund_feateng.csv') # original dataset, used for validation

df_test = df_test[df_test.MT == 16] # extract (n,2n) only
df = df[df.ERG <= 30.0]
df.index = range(len(df))

df_test = df_test[df_test.ERG <= 30.0]
df_test.index = range(len(df_test)) # re-label indices



all_nuclides = []
for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
	if [j, i] in all_nuclides:
		continue
	else:
		all_nuclides.append([j, i]) # format is [Z, A]

validation_nuclides = [] # list of nuclides used for validation
validation_set_size = 25 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(all_nuclides) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)

def make_train(df: pd.DataFrame) -> tuple[list, list]:
	"""la: lower bound for A
	ua: upper bound for A
	arguments la and ua allow data stratification using A
	df: dataframe to use
	Returns X: X values matrix in shape (nsamples, nfeatures)"""

	Z_train = []
	A_train = []
	S2n_train = []
	S2p_train = []
	# XS_train = []
	# Sp_train = []
	# Sn_train = []
	# BEA_train = []
	# Pairing_train = []
	# Sn_c_train = []  # Sn compound
	# gd_train = []
	# N_train = [] # neutron number
	# bd_train = []
	# Radius_train = []
	# n_gap_erg_train = []
	# n_chem_erg_train = []
	# xs_max_train = []
	# n_rms_radius_train = []
	# octupole_deformation_train = []
	# cat_proton_train = []
	# cat_neutron_train = []
	# cat_double_train = []


	# ME_train = []
	# Z_even_train = []
	# A_even_train = []
	# N_even_train = []

	# Shell_train = []
	# Parity_train = []
	# Spin_train = []
	# Decay_Const_train = []
	# Deform_train = []
	# p_gap_erg_train = []
	# p_chem_erg_train = []
	#
	# p_rms_radius_train = []
	# rms_radius_train = []

	# Daughter nucleus properties
	# Sn_d_train = []
	# Sp_d_train = []
	# S2n_d_train = []
	# Pairing_daughter_train = []
	# Parity_daughter_train = []
	# BEA_daughter_train = []
	# S2p_daughter_train = []
	# Shell_daughter_train = []
	# Decay_daughter_train = []
	# ME_daughter_train = []
	# Radius_daughter_train = []
	# BEA_A_daughter_train = []
	# Spin_daughter_train = []
	# Deform_daughter_train = []

	# Compound nucleus properties
	# Sp_compound_train = []
	# BEA_compound_train = []
	# S2n_compound_train = []
	# S2p_compound_train = []
	# Decay_compound_train = []
	# Sn_compound_train = []
	# Shell_compound_train = []
	# Spin_compound_train = []
	# Radius_compound_train = []
	# Deform_compound_train = []
	# ME_compound_train = []
	# BEA_A_compound_train = []
	# Pairing_compound_train = []
	# Parity_compound_train = []




	##########
	# Target value:

	XS_avg_train = []

	for nuc in all_nuclides:  # MT = 16 is (n,2n) (already extracted)
		if nuc in validation_nuclides:
			continue # prevents loop from adding test isotope to training data

		df_temporary = df[df.Z == nuc[0]]
		df_temporary = df_temporary[df_temporary.A == nuc[1]] # isolates nuclide in current iteration
		df_temporary.index = range(len(df_temporary))

		mean_XS = np.mean(df_temporary['XS'])

		Z_train.append(df_temporary['Z'][0])
		A_train.append(df_temporary['A'][0])
		S2n_train.append(df_temporary['S2n'][0])
		S2p_train.append(df_temporary['S2p'][0])
		# Energy_train.append(df_temporary['ERG'][0])
		# Sp_train.append(df_temporary['Sp'][0])
		# Sn_train.append(df_temporary['Sn'][0])
		# BEA_train.append(df_temporary['BEA'][0])
		# Pairing_train.append(df_temporary['Pairing'][0])
		# Sn_c_train.append(Sn_compound[idx])
		# gd_train.append(gamma_deformation[idx])
		# N_train.append(N[idx])
		# bd_train.append(beta_deformation[idx])

		XS_avg_train.append(mean_XS)
		# Sn_d_train.append(Sn_daughter[idx])
		# Sp_d_train.append(Sp_daughter[idx])
		# S2n_d_train.append(S2n_daughter[idx])
		# Radius_train.append(Radius[idx])
		# n_gap_erg_train.append(n_gap_erg[idx])
		# n_chem_erg_train.append(n_chem_erg[idx])
		# Pairing_daughter_train.append(Pairing_daughter[idx])
		# # Parity_daughter_train.append(Parity_daughter[idx])
		# # xs_max_train.append(xs_max[idx])
		# n_rms_radius_train.append(n_rms_radius[idx])
		# # octupole_deformation_train.append(octupole_deformation[idx])
		# Decay_compound_train.append(Decay_compound[idx])
		# BEA_daughter_train.append(BEA_daughter[idx])
		# BEA_compound_train.append(BEA_compound[idx])
		# S2n_compound_train.append(S2n_compound[idx])
		# S2p_compound_train.append(S2p_compound[idx])
		# ME_train.append(ME[idx])
		# # Z_even_train.append(Z_even[idx])
		# # A_even_train.append(A_even[idx])
		# # N_even_train.append(N_even[idx])
		# Shell_train.append(Shell[idx])
		# Parity_train.append(Parity[idx])
		# Spin_train.append(Spin[idx])
		# Decay_Const_train.append(Decay_Const[idx])
		# Deform_train.append(Deform[idx])
		# p_gap_erg_train.append(p_gap_erg[idx])
		# p_chem_erg_train.append(p_chem_erg[idx])
		# p_rms_radius_train.append(p_rms_radius[idx])
		# rms_radius_train.append(rms_radius[idx])
		# Sp_compound_train.append(Sp_compound[idx])
		# Sn_compound_train.append(Sn_compound[idx])
		# Shell_compound_train.append(Shell_compound[idx])
		# # S2p_daughter_train.append(S2p_daughter[idx])
		# Shell_daughter_train.append(Shell_daughter[idx])
		# Spin_compound_train.append(Spin_compound[idx])
		# Radius_compound_train.append(Radius_compound[idx])
		# Deform_compound_train.append(Deform_compound[idx])
		# ME_compound_train.append(ME_compound[idx])
		# BEA_A_compound_train.append(BEA_A_compound[idx])
		# Decay_daughter_train.append(Decay_daughter[idx])
		# ME_daughter_train.append(ME_daughter[idx])
		# Radius_daughter_train.append(Radius_daughter[idx])
		# Pairing_compound_train.append(Pairing_compound[idx])
		# Parity_compound_train.append(Parity_compound[idx])
		# BEA_A_daughter_train.append(BEA_A_daughter[idx])
		# # Spin_daughter_train.append(Spin_daughter[idx])
		# Deform_daughter_train.append(Deform_daughter[idx])
		# cat_proton_train.append(magic_p[idx])
		# cat_neutron_train.append(magic_n[idx])
		# cat_double_train.append(magic_d[idx])
	X = np.array([Z_train, A_train, S2n_train, S2p_train,
				  # Sp_train, Sn_train, BEA_train, Pairing_train, Sn_c_train,
				  # gd_train, N_train, bd_train, Sn_d_train,
				  # Sp_d_train,
				  # S2n_d_train,
				  # Radius_train,
				  # n_gap_erg_train, n_chem_erg_train, #
				  # xs_max_train,
				  # n_rms_radius_train,
				  # octupole_deformation_train,
				  # Decay_compound_train,
				  # BEA_daughter_train,
				  # BEA_compound_train,
				  # Pairing_daughter_train,
				  # Parity_daughter_train,
				  # S2n_compound_train,
				  # S2p_compound_train,
				  # ME_train,
				  # Z_even_train,
				  # A_even_train,
				  # N_even_train,
				  # Shell_train,
				  # Parity_train,
				  # Spin_train,
				  # Decay_Const_train,
				  # Deform_train,
				  # p_gap_erg_train,
				  # p_chem_erg_train,
				  # p_rms_radius_train,
				  # rms_radius_train,
				  # Sp_compound_train,
				  # Sn_compound_train,
				  # Shell_compound_train,
				  # S2p_daughter_train,
				  # Shell_daughter_train,
				  # Spin_compound_train,
				  # Radius_compound_train,
				  # Deform_compound_train,
				  # ME_compound_train,
				  # BEA_A_compound_train,
				  # Decay_daughter_train,
				  # ME_daughter_train,
				  # Radius_daughter_train,
				  # Pairing_compound_train,
				  # Parity_compound_train,
				  # BEA_A_daughter_train,
				  # Spin_daughter_train,
				  # Deform_daughter_train,
				  # cat_proton_train,
				  # cat_neutron_train,
				  # cat_double_train,
				  ])

	

	y = np.array(XS_avg_train)

	X = np.transpose(X) # forms matrix into correct shape (values, features)
	return X, y


def make_test(nuclides: list, df: pd.DataFrame):
	"""
	nuclides: array of (1x2) arrays containing Z of nuclide at index 0, and A at index 1.
	df: dataframe used for validation data

	Returns: xtest - matrix of values for the model to use for making predictions
	ytest: cross sections
	"""

	ztest = [nuclide[0] for nuclide in nuclides] # first element is the Z-value of the given test nuclide
	atest = [nuclide[1] for nuclide in nuclides]

	Z_test = []
	A_test = []
	S2n_test = []
	S2p_test = []
	# XS_train = []
	# Sp_train = []
	# Sn_train = []
	# BEA_train = []
	# Pairing_train = []
	# Sn_c_train = []  # Sn compound
	# gd_train = []
	# N_train = [] # neutron number
	# bd_train = []
	# Radius_train = []
	# n_gap_erg_train = []
	# n_chem_erg_train = []
	# xs_max_train = []
	# n_rms_radius_train = []
	# octupole_deformation_train = []
	# cat_proton_train = []
	# cat_neutron_train = []
	# cat_double_train = []

	# ME_train = []
	# Z_even_train = []
	# A_even_train = []
	# N_even_train = []

	# Shell_train = []
	# Parity_train = []
	# Spin_train = []
	# Decay_Const_train = []
	# Deform_train = []
	# p_gap_erg_train = []
	# p_chem_erg_train = []
	#
	# p_rms_radius_train = []
	# rms_radius_train = []

	# Daughter nucleus properties
	# Sn_d_train = []
	# Sp_d_train = []
	# S2n_d_train = []
	# Pairing_daughter_train = []
	# Parity_daughter_train = []
	# BEA_daughter_train = []
	# S2p_daughter_train = []
	# Shell_daughter_train = []
	# Decay_daughter_train = []
	# ME_daughter_train = []
	# Radius_daughter_train = []
	# BEA_A_daughter_train = []
	# Spin_daughter_train = []
	# Deform_daughter_train = []

	# Compound nucleus properties
	# Sp_compound_train = []
	# BEA_compound_train = []
	# S2n_compound_train = []
	# S2p_compound_train = []
	# Decay_compound_train = []
	# Sn_compound_train = []
	# Shell_compound_train = []
	# Spin_compound_train = []
	# Radius_compound_train = []
	# Deform_compound_train = []
	# ME_compound_train = []
	# BEA_A_compound_train = []
	# Pairing_compound_train = []
	# Parity_compound_train = []

	##########
	# Target value:

	XS_avg_test = []

	for nuc in validation_nuclides:

		df_temporary = df[df.Z == nuc[0]]
		df_temporary = df_temporary[df_temporary.A == nuc[1]]  # isolates nuclide in current iteration
		df_temporary.index = range(len(df_temporary))

		mean_XS = np.mean(df_temporary['XS'])


		Z_test.append(df_temporary['Z'][0])
		A_test.append(df_temporary['A'][0])
		S2n_test.append(df_temporary['S2n'][0])
		S2p_test.append(df_temporary['S2p'][0])

		XS_avg_test.append(mean_XS)
	# XS_test.append(XS[j])
	# Sp_test.append(Sep_p[j])
	# Sn_test.append(Sep_n[j])
	# BEA_test.append(BEA[j])
	# Pairing_test.append(Pairing[j])
	# Sn_c_test.append(Sn_compound[j])
	# gd_test.append(gamma_deformation[j])
	# N_test.append(N[j])
	# bd_test.append(beta_deformation[j])
	# Sn_d_test.append(Sn_daughter[j])
	# Sp_d_test.append(Sp_daughter[j])
	# S2n_d_test.append(S2n_daughter[j])
	# Radius_test.append(Radius[j])
	# n_gap_erg_test.append(n_gap_erg[j])
	# n_chem_erg_test.append(n_chem_erg[j])
	# Pairing_daughter_test.append(Pairing_daughter[j])
	# # xs_max_test.append(np.nan) # cheat feature - nan
	# # octupole_deformation_test.append(octupole_deformation[j])
	#
	# # Parity_daughter_test.append(Parity_daughter[j])
	# n_rms_radius_test.append(n_rms_radius[j])
	# Decay_compound_test.append(Decay_compound[j]) # D_c
	# BEA_daughter_test.append(BEA_daughter[j])
	#
	# BEA_compound_test.append(BEA_compound[j])
	#
	# S2n_compound_test.append(S2n_compound[j])
	# S2p_compound_test.append(S2p_compound[j])
	#
	# ME_test.append(ME[j])
	# # Z_even_test.append(Z_even[j])
	# # A_even_test.append(A_even[j])
	# # N_even_test.append(N_even[j])
	# Shell_test.append(Shell[j])
	# Parity_test.append(Parity[j])
	# Spin_test.append(Spin[j])
	# Decay_Const_test.append(Decay_Const[j])
	# Deform_test.append(Deform[j])
	# p_gap_erg_test.append(p_gap_erg[j])
	# p_chem_erg_test.append(p_chem_erg[j])
	# p_rms_radius_test.append(p_rms_radius[j])
	# rms_radius_test.append(rms_radius[j])
	# Sp_compound_test.append(Sp_compound[j])
	# Sn_compound_test.append(Sn_compound[j])
	# Shell_compound_test.append(Shell_compound[j])
	# # S2p_daughter_test.append(S2p_daughter[j])
	# Shell_daughter_test.append(Shell_daughter[j])
	# Spin_compound_test.append(Spin_compound[j])
	# Radius_compound_test.append(Radius_compound[j])
	# Deform_compound_test.append(Deform_compound[j])
	# ME_compound_test.append(ME_compound[j])
	# BEA_A_compound_test.append(BEA_A_compound[j])
	# Decay_daughter_test.append(Decay_daughter[j])
	# ME_daughter_test.append(ME_daughter[j])
	# Radius_daughter_test.append(Radius_daughter[j])
	# Pairing_compound_test.append(Pairing_compound[j])
	# Parity_compound_test.append(Parity_compound[j])
	# BEA_A_daughter_test.append(BEA_A_daughter[j])
	# # Spin_daughter_test.append(Spin_daughter[j])
	# Deform_daughter_test.append(Deform_daughter[j])
	# cat_proton_test.append(magic_p[j])
	# cat_neutron_test.append(magic_n[j])
	# cat_double_test.append(magic_d[j])




	xtest = np.array([Z_test, A_test, S2n_test, S2p_test,
					  # Energy_test,
					  # Sp_test, Sn_test, BEA_test, Pairing_test, Sn_c_test,
					  # gd_test, N_test, bd_test,
					  # Sn_d_test,
					  # Sp_d_test,
					  # S2n_d_test,
					  # Radius_test,
					  # n_gap_erg_test,
					  # n_chem_erg_test,
					  # # xs_max_test,
					  # n_rms_radius_test,
					  # # octupole_deformation_test,
					  # Decay_compound_test,
					  # BEA_daughter_test,
					  # BEA_compound_test,
					  # Pairing_daughter_test,
					  # # Parity_daughter_test,
					  # S2n_compound_test,
					  # S2p_compound_test,
					  # ME_test,
					  # # Z_even_test,
					  # # A_even_test,
					  # # N_even_test,
					  # Shell_test,
					  # Parity_test,
					  # Spin_test,
					  # Decay_Const_test,
					  # Deform_test,
					  # p_gap_erg_test,
					  # p_chem_erg_test,
					  # p_rms_radius_test,
					  # rms_radius_test,
					  # Sp_compound_test,
					  # Sn_compound_test,
					  # Shell_compound_test,
					  # # S2p_daughter_test,
					  # Shell_daughter_test,
					  # Spin_compound_test,
					  # Radius_compound_test,
					  # Deform_compound_test,
					  # ME_compound_test,
					  # BEA_A_compound_test,
					  # Decay_daughter_test,
					  # ME_daughter_test,
					  # Radius_daughter_test,
					  # Pairing_compound_test,
					  # Parity_compound_test,
					  # BEA_A_daughter_test,
					  # # Spin_daughter_test,
					  # Deform_daughter_test,
					  # cat_proton_test,
					  # cat_neutron_test,
					  # cat_double_test,
					  ])
	xtest = np.transpose(xtest)

	y_test = XS_avg_test

	return xtest, y_test


if __name__ == "__main__":

	X_train, y_train = make_train(df=df) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=df_test)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Data prep done")

	model = xg.XGBRegressor(n_estimators = 850,
							max_leaves = 0,
							max_depth=9,
							learning_rate=0.07)

	time1 = time.time()
	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions

	for true, pred, nuc in zip(y_test, predictions, validation_nuclides):
		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} True: {true:0.4f} b Prediction: {pred:0.4f} b %Error: {((pred - true)/true) *100:0.1f}% \n")
	print()

	print(f"R2: {r2_score(y_test, predictions)}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1} s')

	# explainer = shap.Explainer(model.predict, X_train,
	# 						   feature_names= ['Z', 'A', 'S2n', 'S2p', 'E', 'Sp', 'Sn',
	# 										   'BEA', 'P', 'Snc', 'g-def', 'N',
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
	# 										   'Pair_d',
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
	# 										   'S_n_c',
	# 										   'Shell_c',
	# 										   # 'S2p-d',
	# 										   'Shell-d',
	# 										   'Spin-c',
	# 										   'Rad-c',
	# 										   'Def-c',
	# 										   'ME-c',
	# 										   'BEA-A-c',
	# 										   'Decay-d',
	# 										   'ME-d',
	# 										   'Rad-d',
	# 										   'Pair-c',
	# 										   'Par-c',
	# 										   'BEA-A-d',
	# 										   # 'Spin-d',
	# 										   'Def-d',
	# 										   'mag_p',
	# 										   'mag_n',
	# 										   'mag_d',
	# 										   ]) # SHAP feature importance analysis
	# # shap_values = explainer(X_test)
	#
	# # name features for FIA
	# model.get_booster().feature_names = ['Z', 'A', 'S2n', 'S2p', 'E', 'Sp', 'Sn',
	# 									 'BEA', 'P', 'Snc', 'g-def',
	# 									 'N',
	# 									 'b-def',
	# 									 'Sn da',
	# 									 'Sp d',
	# 									 'S2n d',
	# 									 'Radius',
	# 									 'n_g_erg',
	# 									 'n_c_erg',
	# 									 # 'xsmax',
	# 									 'n_rms_r',
	# 									 # 'oct_def',
	# 									 'D_c',
	# 									 'BEA_d',
	# 									 'BEA_c',
	# 									 'Pair_d',
	# 									 # 'Par_d',
	# 									 'S2n_c',
	# 									 'S2p_c',
	# 									 'ME',
	# 									 # 'Z_even',
	# 									 # 'A_even',
	# 									 # 'N_even',
	# 									 'Shell',
	# 									 'Par',
	# 									 'Spin',
	# 									 'Decay',
	# 									 'Deform',
	# 									 'p_g_e',
	# 									 'p_c_e',
	# 									 'p_rms_r',
	# 									 'rms_r',
	# 									 'Sp_c',
	# 									 'Sn_c',
	# 									 'Shell_c',
	# 									 # 'S2p-d',
	# 									 'Shell-d',
	# 									 'Spin-c',
	# 									 'Rad-c',
	# 									 'Def-c',
	# 									 'ME-c',
	# 									 'BEA-A-c',
	# 									 'Decay-d',
	# 									 'ME-d',
	# 									 'Rad-d',
	# 									 'Pair-c',
	# 									 'Par-c',
	# 									 'BEA-A-d',
	# 									 # 'Spin-d',
	# 									 'Def-d',
	# 									 'mag_p',
	# 									 'mag-n',
	# 									 'mag-d',
	# 									 ]

	# plt.figure(figsize=(10,12))
	# xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60) # metric is total gain
	# plt.show()

	# shap.plots.bar(shap_values, max_display = 70) # display SHAP results
