import pandas as pd

ENDFBVIII_file = pd.read_csv('ENDFBVIII_MT16_uncertainties.csv')
s2x_file = pd.read_csv('S2x_csv_data.csv')

new_columns = ['S2N_U', 'S2P_U']

merged_columns = ['Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'Z', 'A',
       'N', 'ME', 'BEA', 'BEA_A', 'AM', 'Sn', 'S2n', 'Sp', 'S2p', 'Radius',
       'Pairing', 'Shell', 'Spin', 'Parity', 'Deform', 'Decay_Const', 'Z_even',
       'A_even', 'N_even', 'beta_deformation', 'gamma_deformation',
       'octopole_deformation', 'rms_radius', 'n_rms_radius', 'p_rms_radius',
       'n_gap_erg', 'p_gap_erg', 'n_chem_erg', 'p_chem_erg', 'Sn_compound',
       'S2n_compound', 'Sp_compound', 'S2p_compound', 'Radius_compound',
       'ME_compound', 'BEA_compound', 'BEA_A_compound', 'Sn_daughter',
       'S2n_daughter', 'Sp_daughter', 'S2p_daughter', 'Radius_daughter',
       'ME_daughter', 'BEA_daughter', 'BEA_A_daughter', 'Pairing_compound',
       'Shell_compound', 'Spin_compound', 'Parity_compound', 'Deform_compound',
       'Decay_compound', 'Pairing_daughter', 'Shell_daughter', 'Spin_daughter',
       'Parity_daughter', 'Deform_daughter', 'Decay_daughter', 'ERG', 'XS',
       'Nlow', 'Ulow', 'Ntop', 'Utop', 'ainf', 'XSupp', 'XSlow', 'Asymmetry',
       'Asymmetry_compound', 'Asymmetry_daughter', 'unc_r', 'unc_hls',
       'unc_sn', 'unc_sp', 'unc_ba', 'atomic_mass', 'unc_am', 'massexcess',
       'unc_me', 'S2N_U', 'S2P_U']


ENDFBVIII_copy = pd.DataFrame(columns = merged_columns)
for i, row in ENDFBVIII_file.iterrows():
	print(f"Row {i}/{len(ENDFBVIII_file)}")
	target_nuclide = [row['Z'], row['A']]
	for j, data in s2x_file.iterrows():
		if [data['Z'], data['A']] == target_nuclide:
			copy_row = row
			for idx, name in enumerate(new_columns):
				copy_row[name] = data[name]

			ENDFBVIII_copy = ENDFBVIII_copy._append(copy_row, ignore_index=True)
