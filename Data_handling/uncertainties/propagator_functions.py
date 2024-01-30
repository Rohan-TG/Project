import random

def training_sampler(df, target_nuclides, LA, UA, sampled_uncertainties, exclusions = []):
	""""""

	ME = df['ME']
	Z = df['Z']
	A = df['A']
	Sep_2n = df['S2n']
	Sep_2p = df['S2p']
	Energy = df['ERG']
	XS = df['XS']
	Sep_p = df['Sp']
	Sep_n = df['Sn']
	BEA = df['BEA']
	# Pairing = df['Pairing']
	gamma_deformation = df['gamma_deformation']
	beta_deformation = df['beta_deformation']
	# octupole_deformation = df['octopole_deformation']
	# Z_even = df['Z_even']
	# A_even = df['A_even']
	# N_even = df['N_even']
	N = df['N']
	# xs_max = df['xs_max']
	Radius = df['Radius']
	Shell = df['Shell']
	Parity = df['Parity']
	Spin = df['Spin']
	Decay_Const = df['Decay_Const']
	Deform = df['Deform']
	n_gap_erg = df['n_gap_erg']
	n_chem_erg = df['n_chem_erg']
	p_gap_erg = df['p_gap_erg']
	p_chem_erg = df['p_chem_erg']
	n_rms_radius = df['n_rms_radius']
	p_rms_radius = df['p_rms_radius']
	# rms_radius = df['rms_radius']
	# magic_p = df['cat_magic_proton']
	# magic_n = df['cat_magic_neutron']
	# magic_d = df['cat_magic_double']
	Nlow = df['Nlow']
	Ulow = df['Ulow']
	# Ntop = df['Ntop']
	Utop = df['Utop']
	ainf = df['ainf']
	# XSlow = df['XSlow']
	# XSupp = df['XSupp']
	Asymmetry = df['Asymmetry']

	# Compound nucleus properties
	Sp_compound = df['Sp_compound']
	Sn_compound = df['Sn_compound']
	S2n_compound = df['S2n_compound']
	S2p_compound = df['S2p_compound']
	Decay_compound = df['Decay_compound']
	BEA_compound = df['BEA_compound']
	BEA_A_compound = df['BEA_A_compound']
	Radius_compound = df['Radius_compound']
	# ME_compound = df['ME_compound']
	# Pairing_compound = df['Pairing_compound']
	Shell_compound = df['Shell_compound']
	Spin_compound = df['Spin_compound']
	# Parity_compound = df['Parity_compound']
	Deform_compound = df['Deform_compound']
	Asymmetry_compound = df['Asymmetry_compound']

	# Daughter nucleus properties
	Sn_daughter = df['Sn_daughter']
	S2n_daughter = df['S2n_daughter']
	Sp_daughter = df['Sp_daughter']
	# Pairing_daughter = df['Pairing_daughter']
	# Parity_daughter = df['Parity_daughter']
	BEA_daughter = df['BEA_daughter']
	# Shell_daughter = df['Shell_daughter']
	# S2p_daughter = df['S2p_daughter']
	# Radius_daughter = df['Radius_daughter']
	ME_daughter = df['ME_daughter']
	BEA_A_daughter = df['BEA_A_daughter']
	# Spin_daughter = df['Spin_daughter']
	Deform_daughter = df['Deform_daughter']
	Decay_daughter = df['Decay_daughter']
	Asymmetry_daughter = df['Asymmetry_daughter']


	### UNCERTAINTY DATA
	Sn_uncertainties = df['unc_sn']


	Z_train = []
	A_train = []
	S2n_train = []
	S2p_train = []
	Energy_train = []
	XS_train = []
	Sp_train = []
	Sn_train = []
	BEA_train = []
	# Pairing_train = []
	gd_train = []
	N_train = [] # neutron number
	bd_train = []
	Radius_train = []
	n_gap_erg_train = []
	n_chem_erg_train = []
	# xs_max_train = []
	n_rms_radius_train = []
	# octupole_deformation_train = []
	# cat_proton_train = []
	# cat_neutron_train = []
	# cat_double_train = []
	ME_train = []
	# Z_even_train = []
	# A_even_train = []
	# N_even_train = []
	Shell_train = []
	Parity_train = []
	Spin_train = []
	Decay_Const_train = []
	Deform_train = []
	p_gap_erg_train = []
	p_chem_erg_train = []
	p_rms_radius_train = []
	# rms_radius_train = []
	Nlow_train = []
	Ulow_train = []
	# Ntop_train = []
	Utop_train = []
	ainf_train = []
	# XSlow_train = []
	# XSupp_train = []
	Asymmetry_train = []



	# Daughter nucleus properties
	Sn_d_train = []
	Sp_d_train = []
	S2n_d_train = []
	# Pairing_daughter_train = []
	# Parity_daughter_train = []
	BEA_daughter_train = []
	# S2p_daughter_train = []
	# Shell_daughter_train = []
	Decay_daughter_train = []
	ME_daughter_train = []
	# Radius_daughter_train = []
	BEA_A_daughter_train = []
	# Spin_daughter_train = []
	Deform_daughter_train = []
	Asymmetry_daughter_train = []

	# Compound nucleus properties
	Sn_c_train = []  # Sn compound
	Sp_compound_train = []
	BEA_compound_train = []
	S2n_compound_train = []
	S2p_compound_train = []
	Decay_compound_train = []
	# Sn_compound_train = []
	Shell_compound_train = []
	Spin_compound_train = []
	Radius_compound_train = []
	Deform_compound_train = []
	ME_compound_train = []
	BEA_A_compound_train = []
	# Pairing_compound_train = []
	# Parity_compound_train = []
	Asymmetry_compound_train = []


	for idx, unused in enumerate(Z):  # MT = 16 is (n,2n) (already extracted)
		if [Z[idx], A[idx]] in target_nuclides:
			continue # prevents loop from adding test isotope data to training data
		if [Z[idx], A[idx]] in exclusions:
			continue
		if Energy[idx] > 20: # training on data less than 30 MeV
			continue
		if A[idx] <= UA and A[idx] >= LA: # checks that nuclide is within bounds for A
			Z_train.append(Z[idx])
			A_train.append(A[idx])
			S2n_train.append(Sep_2n[idx])
			S2p_train.append(Sep_2p[idx])
			Energy_train.append(Energy[idx])
			XS_train.append(XS[idx])
			Sp_train.append(Sep_p[idx])

			BEA_train.append(BEA[idx])
			# Pairing_train.append(Pairing[idx])
			Sn_c_train.append(Sn_compound[idx])
			gd_train.append(gamma_deformation[idx])
			N_train.append(N[idx])
			bd_train.append(beta_deformation[idx])
			Sn_d_train.append(Sn_daughter[idx])
			Sp_d_train.append(Sp_daughter[idx])
			S2n_d_train.append(S2n_daughter[idx])
			Radius_train.append(Radius[idx])
			n_gap_erg_train.append(n_gap_erg[idx])
			n_chem_erg_train.append(n_chem_erg[idx])
			# Pairing_daughter_train.append(Pairing_daughter[idx])
			# Parity_daughter_train.append(Parity_daughter[idx])
			# xs_max_train.append(xs_max[idx])
			n_rms_radius_train.append(n_rms_radius[idx])
			# octupole_deformation_train.append(octupole_deformation[idx])
			Decay_compound_train.append(Decay_compound[idx])
			BEA_daughter_train.append(BEA_daughter[idx])
			BEA_compound_train.append(BEA_compound[idx])
			S2n_compound_train.append(S2n_compound[idx])
			S2p_compound_train.append(S2p_compound[idx])
			ME_train.append(ME[idx])
			# Z_even_train.append(Z_even[idx])
			# A_even_train.append(A_even[idx])
			# N_even_train.append(N_even[idx])
			Shell_train.append(Shell[idx])
			Parity_train.append(Parity[idx])
			Spin_train.append(Spin[idx])
			Decay_Const_train.append(Decay_Const[idx])
			Deform_train.append(Deform[idx])
			p_gap_erg_train.append(p_gap_erg[idx])
			p_chem_erg_train.append(p_chem_erg[idx])
			p_rms_radius_train.append(p_rms_radius[idx])
			# rms_radius_train.append(rms_radius[idx])
			Sp_compound_train.append(Sp_compound[idx])
			# Sn_compound_train.append(Sn_compound[idx])
			Shell_compound_train.append(Shell_compound[idx])
			# S2p_daughter_train.append(S2p_daughter[idx])
			# Shell_daughter_train.append(Shell_daughter[idx])
			Spin_compound_train.append(Spin_compound[idx])
			Radius_compound_train.append(Radius_compound[idx])
			Deform_compound_train.append(Deform_compound[idx])
			# ME_compound_train.append(ME_compound[idx])
			BEA_A_compound_train.append(BEA_A_compound[idx])
			Decay_daughter_train.append(Decay_daughter[idx])
			ME_daughter_train.append(ME_daughter[idx])
			# Radius_daughter_train.append(Radius_daughter[idx])
			# Pairing_compound_train.append(Pairing_compound[idx])
			# Parity_compound_train.append(Parity_compound[idx])
			BEA_A_daughter_train.append(BEA_A_daughter[idx])
			# Spin_daughter_train.append(Spin_daughter[idx])
			Deform_daughter_train.append(Deform_daughter[idx])
			Nlow_train.append(Nlow[idx])
			Ulow_train.append(Ulow[idx])
			# Ntop_train.append(Ntop[idx])
			Utop_train.append(Utop[idx])
			ainf_train.append(ainf[idx])
			# XSlow_train.append(XSlow[idx])
			# XSupp_train.append(XSupp[idx])
			Asymmetry_train.append(Asymmetry[idx])
			Asymmetry_compound_train.append(Asymmetry_compound[idx])
			Asymmetry_daughter_train.append(Asymmetry_daughter[idx])
			# AM_train.append(AM[idx])

		### UNCERTAINTIES
		uncertainty_value = Sn_uncertainties[idx]


		Sn_train.append(Sep_n[idx])

	pass