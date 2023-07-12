import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import periodictable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score
# import george



df = pd.read_csv('zeroed_1_xs_fund_feateng.csv') # new interpolated dataset, used for training only
df_test = pd.read_csv('zeroed_1_xs_fund_feateng.csv') # original dataset, used for validation

df_test = df_test[df_test.MT == 16] # extract (n,2n) only
df_test.index = range(len(df_test)) # re-label indices


def anomaly_remover(dfa):
	anomalies = [[14, 28], [14, 29], [14, 30], [15, 31], [20, 40], [20, 42], [20, 43], [20, 44], [20, 46],
				 [20, 48], [24, 50], [24, 52], [24, 53], [24, 54], [28, 58], [28, 60], [28, 61], [28, 62], [28, 64],
				 [29, 63], [29, 65], [41, 93], [80, 196], [80, 198], [80, 199], [80, 200], [80, 201], [80, 202],
				 [80, 204], [82, 206], [82, 207]]
	"""list of nuclides with cutoff value @ 20 MeV - format is [Z,A]"""

	anomaly_indices = []
	for set in anomalies:
		z = set[0]
		a = set[1]

		remover = dfa[dfa.A == a]
		remove = remover[remover.Z == z]

		last_row_index = remove.index[-1]
		anomaly_indices.append(last_row_index)


	for i in anomaly_indices:
		dfa = dfa.drop(index=i)

	dfa.index = range(len(dfa))

	return dfa

df_test = anomaly_remover(dfa = df_test)

al = []
for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
	if [j, i] in al:
		continue
	else:
		al.append([j, i]) # format is [Z, A]


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

validation_nuclides = [] # list of nuclides used for validation
validation_set_size = 1 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(al) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")

def make_train(df, la=25, ua=210):
	"""la: lower bound for A
	ua: upper bound for A
	arguments la and ua allow data stratification using A
	df: dataframe to use
	Returns X: X values matrix in shape (nsamples, nfeatures)"""

	MT = df['MT']
	# ME = df['ME']
	Z = df['Z']
	A = df['A']
	Sep_2n = df['S2n']
	Sep_2p = df['S2p']
	Energy = df['ERG']
	XS = df['XS']
	Sep_p = df['Sp']
	Sep_n = df['Sn']
	N = df['N']
	# BEA = df['BEA']
	# Pairing = df['Pairing']
	# gamma_deformation = df['gamma_deformation']
	# beta_deformation = df['beta_deformation']
	# octupole_deformation = df['octopole_deformation']
	# Z_even = df['Z_even']
	# A_even = df['A_even']
	# N_even = df['N_even']

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
	# magic_p = df['cat_magic_proton']
	# magic_n = df['cat_magic_neutron']
	# magic_d = df['cat_magic_double']

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

	XS_train = []

	X = []

	for idx, r_type in enumerate(MT):  # MT = 16 is (n,2n) (already extracted)
		working_array = []
		if [Z[idx], A[idx]] in validation_nuclides:
			continue # prevents loop from adding test isotope to training data
		if Energy[idx] >= 30: # training on data less than 30 MeV
			continue
		if A[idx] <= ua and A[idx] >= la: # checks that nuclide is within bounds for A

			# working_array.append(Z[idx])
			# working_array.append(A[idx])
			# working_array.append(Sep_2n[idx])
			# working_array.append(Sep_2p[idx])
			working_array.append(Energy[idx])
			# working_array.append(Sep_p[idx])
			# working_array.append(Sep_n[idx])
			# working_array.append(N[idx])

			XS_train.append(XS[idx])

			X.append(working_array)
			# Z_train.append(Z[idx])
			# A_train.append(A[idx])
			# S2n_train.append(Sep_2n[idx])
			# S2p_train.append(Sep_2p[idx])
			# Energy_train.append(Energy[idx])
			# XS_train.append(XS[idx])
			# Sp_train.append(Sep_p[idx])
			# Sn_train.append(Sep_n[idx])
			# N_train.append(N[idx])
			# BEA_train.append(BEA[idx])
			# Pairing_train.append(Pairing[idx])
			# Sn_c_train.append(Sn_compound[idx])
			# gd_train.append(gamma_deformation[idx])

			# bd_train.append(beta_deformation[idx])
			# Sn_d_train.append(Sn_daughter[idx])
			# Sp_d_train.append(Sp_daughter[idx])
			# S2n_d_train.append(S2n_daughter[idx])
			# Radius_train.append(Radius[idx])
			# n_gap_erg_train.append(n_gap_erg[idx])
			# n_chem_erg_train.append(n_chem_erg[idx])
			# Pairing_daughter_train.append(Pairing_daughter[idx])
			# Parity_daughter_train.append(Parity_daughter[idx])
			# xs_max_train.append(xs_max[idx])
			# n_rms_radius_train.append(n_rms_radius[idx])
			# octupole_deformation_train.append(octupole_deformation[idx])
			# Decay_compound_train.append(Decay_compound[idx])
			# BEA_daughter_train.append(BEA_daughter[idx])
			# BEA_compound_train.append(BEA_compound[idx])
			# S2n_compound_train.append(S2n_compound[idx])
			# S2p_compound_train.append(S2p_compound[idx])
			# ME_train.append(ME[idx])
			# Z_even_train.append(Z_even[idx])
			# A_even_train.append(A_even[idx])
			# N_even_train.append(N_even[idx])
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
			# S2p_daughter_train.append(S2p_daughter[idx])
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
			# Spin_daughter_train.append(Spin_daughter[idx])
			# Deform_daughter_train.append(Deform_daughter[idx])
			# cat_proton_train.append(magic_p[idx])
			# cat_neutron_train.append(magic_n[idx])
			# cat_double_train.append(magic_d[idx])


	# X = np.array(X)

	y = XS_train # cross sections

	# X = np.transpose(X) # forms matrix into correct shape (values, features)
	return X, y

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
	Sep_2n = df['S2n']
	Sep_2p = df['S2p']
	Energy = df['ERG']
	XS = df['XS']
	Sep_p = df['Sp']
	Sep_n = df['Sn']
	N = df['N']
	# BEA = df['BEA']
	# Pairing = df['Pairing']
	# gamma_deformation = df['gamma_deformation']
	# beta_deformation = df['beta_deformation']
	# octupole_deformation = df['octopole_deformation']
	# Z_even = df['Z_even']
	# A_even = df['A_even']
	# N_even = df['N_even']

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
	# magic_p = df['cat_magic_proton']
	# magic_n = df['cat_magic_neutron']
	# magic_d = df['cat_magic_double']

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

	XS_test = []

	xtest = []

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		working_array = []
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a and Energy[j] <= 30:

				# working_array.append(Z[j])
				# working_array.append(A[j])
				# working_array.append(Sep_2n[j])
				# working_array.append(Sep_2p[j])
				working_array.append(Energy[j])
				# working_array.append(Sep_p[j])
				# working_array.append(Sep_n[j])
				# working_array.append(N[j])

				XS_test.append(XS[j])

				xtest.append(working_array)


	y_test = XS_test

	return xtest, y_test


X_train, y_train = make_train(df=df) # make training matrix

X_test, y_test = make_test(validation_nuclides, df=df_test)

print("Data prep complete")

kernel = RBF(length_scale='auto')

gaussian_process = GaussianProcessRegressor(kernel=kernel)
print("Regressor generated")

gaussian_process.fit(X_train, y_train)
# gaussian_process.kernel_
print("Training complete")
#
# mean_predictions, std_prediction = gaussian_process.predict(X_test, return_std=True)
#
#
# XS_plotmatrix = []
# E_plotmatrix = []
# P_plotmatrix = []
#
# for nuclide in validation_nuclides:
# 	dummy_test_XS = []
# 	dummy_test_E = []
# 	dummy_predictions = []
# 	for i, row in enumerate(X_test):
# 		if [row[0], row[1]] == nuclide:
# 			dummy_test_XS.append(y_test[i])
# 			dummy_test_E.append(row[4]) # Energy values are in 5th row
# 			dummy_predictions.append(mean_predictions[i])
#
# 	XS_plotmatrix.append(dummy_test_XS)
# 	E_plotmatrix.append(dummy_test_E)
# 	P_plotmatrix.append(dummy_predictions)
#
# # plot predictions against data
# for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
# 	nuc = validation_nuclides[i] # validation nuclide
# 	plt.plot(erg, pred_xs, label='predictions')
# 	plt.plot(erg, true_xs, label='data')
# 	plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
# 	plt.legend()
# 	plt.grid()
# 	plt.ylabel('XS / b')
# 	plt.xlabel('Energy / MeV')
# 	plt.show()
#
# 	r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
# 	print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")