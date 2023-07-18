import time
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import periodictable


# df = pd.read_csv('interpolated_n2_1_xs_fund_feateng.csv') # new interpolated dataset, used for training only
df_test = pd.read_csv('CENDL33_features_unzeroed.csv') # original dataset, used for validation


def make_test(nuclides, df):
	"""
	nuclides: array of (1x2) arrays containing Z of nuclide at index 0, and A at index 1.
	df: dataframe used for validation data

	Returns: xtest - matrix of values for the model to use for making predictions
	ytest: cross sections
	"""

	ztest = [nuclide[0] for nuclide in nuclides] # first element is the Z-value of the given test nuclide
	atest = [nuclide[1] for nuclide in nuclides]


	# MT = df['MT']
	# ME = df['ME']
	Z = df['Z']
	A = df['A']
	# Sep_2n = df['S2n']
	# Sep_2p = df['S2p']
	Energy = df['ERG']
	XS = df['XS']
	# Sep_p = df['Sp']
	# Sep_n = df['Sn']
	# BEA = df['BEA']
	# Pairing = df['Pairing']
	# gamma_deformation = df['gamma_deformation']
	# beta_deformation = df['beta_deformation']
	# octupole_deformation = df['octopole_deformation']
	# Z_even = df['Z_even']
	# A_even = df['A_even']
	# N_even = df['N_even']
	# N = df['N']
	# xs_max = df['xs_max']
	# Radius = df['Radius']
	# Shell = df['Shell']
	# Parity = df['Parity']
	# Spin = df['Spin']
	# Decay_Const = df['Decay_Const']
	# Deform = df['Deform']
	# n_gap_erg = df['n_gap_erg']
	# n_chem_erg = df['n_chem_erg']
	# p_gap_erg = df['p_gap_erg']
	# p_chem_erg = df['p_chem_erg']
	# n_rms_radius = df['n_rms_radius']
	# p_rms_radius = df['p_rms_radius']
	# rms_radius = df['rms_radius']

	# Compound nucleus properties
	# Sp_compound = df['Sp_compound']
	# Sn_compound = df['Sn_compound']
	# S2n_compound = df['S2n_compound']
	# S2p_compound = df['S2p_compound']
	# Decay_compound = df['Decay_compound']
	# BEA_compound = df['BEA_compound']
	# BEA_A_compound = df['BEA_A_compound']
	# Radius_compound = df['Radius_compound']
	# ME_compound = df['ME_compound']
	# Pairing_compound = df['Pairing_compound']
	# Shell_compound = df['Shell_compound']
	# Spin_compound = df['Spin_compound']
	# Parity_compound = df['Parity_compound']
	# Deform_compound = df['Deform_compound']

	# Daughter nucleus properties
	# Sn_daughter = df['Sn_daughter']
	# S2n_daughter = df['S2n_daughter']
	# Sp_daughter = df['Sp_daughter']
	# Pairing_daughter = df['Pairing_daughter']
	# Parity_daughter = df['Parity_daughter']
	# BEA_daughter = df['BEA_daughter']
	# Shell_daughter = df['Shell_daughter']
	# S2p_daughter = df['S2p_daughter']
	# Radius_daughter = df['Radius_daughter']
	# ME_daughter = df['ME_daughter']
	# BEA_A_daughter = df['BEA_A_daughter']
	# Spin_daughter = df['Spin_daughter']
	# Deform_daughter = df['Deform_daughter']
	# Decay_daughter = df['Decay_daughter']



	Z_test = []
	A_test = []
	# Sep2n_test = []
	# Sep2p_test = []
	Energy_test = []
	XS_test = []
	# Sp_test = []
	# Sn_test = []
	# BEA_test = []
	# Pairing_test = []
	# Sn_c_test = []
	# gd_test = [] # gamma deformation
	# N_test = []
	# bd_test = []
	# Radius_test = []
	# n_gap_erg_test = []
	# n_chem_erg_test = []
	# xs_max_test = []
	# octupole_deformation_test = []


	# Daughter features
	# Sn_d_test = []
	# Sp_d_test = []
	# S2n_d_test = []
	# n_rms_radius_test = []
	# Decay_compound_test = []
	# S2p_daughter_test = []
	# Shell_daughter_test = []
	# Decay_daughter_test = []
	# ME_daughter_test = []
	# BEA_daughter_test = []
	# Radius_daughter_test = []
	# BEA_A_daughter_test = []
	# Spin_daughter_test = []
	# Deform_daughter_test = []
	# Pairing_daughter_test = []
	# Parity_daughter_test = []


	# BEA_compound_test = []
	#
	# S2n_compound_test = []
	# S2p_compound_test = []
	# ME_test = []
	# Z_even_test = []
	# A_even_test = []
	# N_even_test = []

	# Shell_test = []
	# Parity_test = []
	# Spin_test = []
	# Decay_Const_test = []
	# Deform_test = []
	# p_gap_erg_test = []
	# p_chem_erg_test = []
	# p_rms_radius_test = []
	# rms_radius_test = []


	# Sp_compound_test = []
	# Sn_compound_test = []
	# Shell_compound_test = []
	# Spin_compound_test = []
	# Radius_compound_test  = []
	# Deform_compound_test = []
	# ME_compound_test = []
	# BEA_A_compound_test = []
	# Pairing_compound_test = []
	# Parity_compound_test = []




	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a and Energy[j] <= 30:
				Z_test.append(Z[j])
				A_test.append(A[j])
				# Sep2n_test.append(Sep_2n[j])
				# Sep2p_test.append(Sep_2p[j])
				Energy_test.append(Energy[j])
				XS_test.append(XS[j])
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
				# xs_max_test.append(np.nan) # cheat feature - nan
				# octupole_deformation_test.append(octupole_deformation[j])

				# Parity_daughter_test.append(Parity_daughter[j])
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
				# Z_even_test.append(Z_even[j])
				# A_even_test.append(A_even[j])
				# N_even_test.append(N_even[j])
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
				# S2p_daughter_test.append(S2p_daughter[j])
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
				# Spin_daughter_test.append(Spin_daughter[j])
				# Deform_daughter_test.append(Deform_daughter[j])

	xtest = np.array([Z_test, A_test,
					  # Sep2n_test, Sep2p_test,
					  Energy_test,
					  # Sp_test, Sn_test, BEA_test, Pairing_test, Sn_c_test,
					  # gd_test, N_test, bd_test,
					  # Sn_d_test,
					  # Sp_d_test,
					  # S2n_d_test,
					  # Radius_test,
					  # n_gap_erg_test,
					  # n_chem_erg_test,
					  # xs_max_test,
					  # n_rms_radius_test,
					  # octupole_deformation_test,
					  # Decay_compound_test,
					  # BEA_daughter_test,
					  # BEA_compound_test,
					  # Pairing_daughter_test,
					  # Parity_daughter_test,
					  # S2n_compound_test,
					  # S2p_compound_test,
					  # ME_test,
					  # Z_even_test,
					  # A_even_test,
					  # N_even_test,
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
					  # S2p_daughter_test,
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
					  # Spin_daughter_test,
					  # Deform_daughter_test,
					  ])
	xtest = np.transpose(xtest)

	y_test = XS_test

	return xtest, y_test



def zero_maker(df):
	"""input df: Dataframe to be zeroed

	Function inserts 0 XS values for energies below reaction threshold.

	Returns: New dataframe with the additional values."""

	df_zeroed = pd.DataFrame(columns=df.columns)  # empty dataframe with column names only, to be appended to
	current_nuclide = [] # iteration nuclide

	for i, row in df.iterrows():
		if i % 1000 == 0:
			print(f"{i}/{len(df['Z'])}")

		if [row['Z'], row['A']] != current_nuclide: # for new nuclide in iteration, insert n number of 0 values

			min_energy = row['ERG'] # Threshold energy
			# energy_app_list = np.linspace(0, min_energy-0.1, 20) # 20 values, 0 to threshold MeV - 0.1
			energy_app_list = np.arange(0.0, min_energy, 0.1)

			dummy_row = row # to be copied
			dummy_row['XS'] = 0.0 # set XS value to 0 for all energies below threshold

			for erg in energy_app_list:
				dummy_row['ERG'] = erg # set energy value
				df_zeroed = df_zeroed._append(dummy_row, ignore_index=True) # append new row

			current_nuclide = [row['Z'],row['A']] # once iteration complete, update the iteration nuclide
		else:
			df_zeroed = df_zeroed._append(row, ignore_index=True) # if below-threshold values have already been added, append normal row

		# print(current_nuclide) # tracker

	df_zeroed.index = range(len(df_zeroed)) # reindex dataframe

	return df_zeroed



df_zero = zero_maker(df=df_test)
print(df_zero.head())

df_zero.to_csv('CENDL33_features_correctly_zeroed.csv')

# to_reduce = pd.read_csv('TENDL21_MT16_XS.csv')
#
# to_reduce = to_reduce[to_reduce.ERG <20.0]
# to_reduce.index = range(len(to_reduce))
#
#
# to_reduce.to_csv('JENDL4_correctly_zeroed_with_features.csv')
# df_zero = pd.read_csv("Third_attempt_TENDL.csv")
# #
# df_using = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
# #
# from matrix_functions import range_setter
# al = range_setter(df=df_using, la=0, ua=260)
#
# TENDL_nuclides = range_setter(df=df_zero, la=0, ua=300)
#
# for nuc in TENDL_nuclides:
#
# 	if nuc in al:
#
# 		X_test, y_test = make_test(nuclides=[nuc], df=df_zero)
#
# 		endfx, endfy = make_test(nuclides=[nuc], df=df_using)
#
# 		plt.figure()
# 		plt.plot(X_test[:,2], y_test, label='TENDL')
# 		plt.grid()
# 		plt.title(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
#
# 		plt.plot(endfx[:,2], endfy, 'x-', label='ENDF/B')
# 		plt.legend()
# 		plt.show()
#
# 		time.sleep(1.5)
