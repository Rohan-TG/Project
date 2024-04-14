import numpy as np

def General_plotter(df, nuclides):
	"""df: dataframe source of XSs
	nuclides: must be array of 1x2 arrays [z,a]

	Returns XS and ERG values. Designed for plotting graphs and doing r2 comparisons without running make_test
	which is much more demanding"""

	ztest = [nuclide[0] for nuclide in nuclides]  # first element is the Z-value of the given test nuclide
	atest = [nuclide[1] for nuclide in nuclides]

	Z = df['Z']
	A = df['A']

	Energy = df['ERG']
	XS = df['XS']

	Z_test = []
	A_test = []
	Energy_test = []
	XS_test = []

	for nuc_test_z, nuc_test_a in zip(ztest, atest):
		for j, (zval, aval) in enumerate(zip(Z, A)):
			if zval == nuc_test_z and aval == nuc_test_a and Energy[j] <= 20:
				Z_test.append(Z[j])
				A_test.append(A[j])
				Energy_test.append(Energy[j])
				XS_test.append(XS[j])

	energies = np.array(Energy_test)

	xs = XS_test

	return energies, xs