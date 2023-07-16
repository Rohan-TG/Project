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
from matrix_functions import anomaly_remover, range_setter



df = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv') # new interpolated dataset, used for training only
df_test = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv') # original dataset, used for validation

df_test = anomaly_remover(dfa = df_test)

al = range_setter(df=df, la=30, ua=215)

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
	# N = df['N']
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

	for idx, unused in enumerate(Z):  # MT = 16 is (n,2n) (already extracted)
		if [Z[idx], A[idx]] in validation_nuclides:
			continue # prevents loop from adding test isotope to training data
		if Energy[idx] >= 30: # training on data less than 30 MeV
			continue
		if A[idx] <= ua and A[idx]:# >= la and Energy[idx] > 0: # checks that nuclide is within bounds for A

	for idx, unused in enumerate(Z):  # MT = 16 is (n,2n) (already extracted)
		if [Z[idx], A[idx]] in validation_nuclides:
			continue # prevents loop from adding test isotope to training data
		if Energy[idx] >= 30: # training on data less than 30 MeV
			continue
		if A[idx] <= ua and A[idx]:# >= la and Energy[idx] > 0: # checks that nuclide is within bounds for A

			X.append(Energy[idx])


			XS_train.append(XS[idx])

	X = np.array(X)

	y = XS_train

	X = X.reshape(-1,1) # forms matrix into correct shape (values, features)
	# print(X)
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
	# Sep_2n = df['S2n']
	# Sep_2p = df['S2p']
	Energy = df['ERG']
	XS = df['XS']
	# Sep_p = df['Sp']
	# Sep_n = df['Sn']
	# N = df['N']
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
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a and Energy[j] < 30:
				# if Energy[j] > 0 and XS[j] > 0:

				# working_array.append(Z[j])
				# working_array.append(A[j])
				# working_array.append(Sep_2n[j])
				# working_array.append(Sep_2p[j])
				xtest.append(Energy[j])
				# working_array.append(Sep_p[j])
				# working_array.append(Sep_n[j])
				# working_array.append(N[j])

				XS_test.append(XS[j])

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
	   for j, (zval, aval) in enumerate(zip(Z, A)):
		  if zval == nuc_test_z and aval == nuc_test_a and Energy[j] < 30:
			 # if Energy[j] > 0 and XS[j] > 0:

			 # working_array.append(Z[j])
			 # working_array.append(A[j])
			 # working_array.append(Sep_2n[j])
			 # working_array.append(Sep_2p[j])
			 xtest.append(Energy[j])
			 # working_array.append(Sep_p[j])
			 # working_array.append(Sep_n[j])
			 # working_array.append(N[j])

			 XS_test.append(XS[j])


	y_test = XS_test
	xtest = np.array(xtest)
	xtest = xtest.reshape(-1,1)
	return xtest, y_test


# X_train, y_train = make_train(df=df) # make training matrix


X_test, y_test = make_test(validation_nuclides, df=df_test)

y_test = np.array(y_test)
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y_test.size), size=10, replace=False)
X_train, y_train = X_test[training_indices], y_test[training_indices]

print("Data prep complete")


kernel = RBF(length_scale=1.0)
gaussian_process = GaussianProcessRegressor(kernel=kernel)
print("Regressor generated")

gaussian_process.fit(X_train,y_train)
print("Training complete")

mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)

# plt.plot(X_test, mean_prediction, label="Mean prediction")
# plt.show()


plt.plot(X_test, y_test, label=r"$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X_test, mean_prediction, label="Mean prediction")
plt.fill_between(
	X_test.ravel(),
	mean_prediction - 1.96 * std_prediction,
	mean_prediction + 1.96 * std_prediction,
	alpha=0.5,
	label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.title(f"{periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.grid()
plt.show()

print(f"r2: {r2_score(y_true=y_test, y_pred=mean_prediction)}")
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

print(f"r2: {r2_score(y_true=y_test, y_pred=mean_prediction)}")
