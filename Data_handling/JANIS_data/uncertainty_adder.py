import pandas as pd
import tqdm

livechart_data = pd.read_csv('livechart_ground_state_data.csv')
ENDFBVIII_file = pd.read_csv('noasym.csv')
# main_file = pd.read_csv('1_fund.csv')





main_columns = ['Z', 'A', 'N', 'ME',
       'BEA', 'BEA_A', 'AM', 'Sn', 'S2n', 'Sp', 'S2p', 'Radius', 'Pairing',
       'Shell', 'Spin', 'Parity', 'Deform', 'Decay_Const', 'Z_even', 'A_even',
       'N_even', 'beta_deformation', 'gamma_deformation',
       'octopole_deformation', 'rms_radius', 'n_rms_radius', 'p_rms_radius',
       'n_gap_erg', 'p_gap_erg', 'n_chem_erg', 'p_chem_erg', 'Sn_compound',
       'S2n_compound', 'Sp_compound', 'S2p_compound', 'Radius_compound',
       'ME_compound', 'BEA_compound', 'BEA_A_compound', 'Sn_daughter',
       'S2n_daughter', 'Sp_daughter', 'S2p_daughter', 'Radius_daughter',
       'ME_daughter', 'BEA_daughter', 'BEA_A_daughter', 'Pairing_compound',
       'Shell_compound', 'Spin_compound', 'Parity_compound', 'Deform_compound',
       'Decay_compound', 'Pairing_daughter', 'Shell_daughter', 'Spin_daughter',
       'Parity_daughter', 'Deform_daughter', 'Decay_daughter', 'ERG', 'XS', 'dXS'
		]

new_cols = ['unc_r', 'unc_hls', 'unc_sn', 'unc_sp', 'unc_ba', 'atomic_mass', 'unc_am', 'massexcess',
			'unc_me',]

merged_columns = ['Z', 'A', 'N', 'ME',
       'BEA', 'BEA_A', 'AM', 'Sn', 'S2n', 'Sp', 'S2p', 'Radius', 'Pairing',
       'Shell', 'Spin', 'Parity', 'Deform', 'Decay_Const', 'Z_even', 'A_even',
       'N_even', 'beta_deformation', 'gamma_deformation',
       'octopole_deformation', 'rms_radius', 'n_rms_radius', 'p_rms_radius',
       'n_gap_erg', 'p_gap_erg', 'n_chem_erg', 'p_chem_erg', 'Sn_compound',
       'S2n_compound', 'Sp_compound', 'S2p_compound', 'Radius_compound',
       'ME_compound', 'BEA_compound', 'BEA_A_compound', 'Sn_daughter',
       'S2n_daughter', 'Sp_daughter', 'S2p_daughter', 'Radius_daughter',
       'ME_daughter', 'BEA_daughter', 'BEA_A_daughter', 'Pairing_compound',
       'Shell_compound', 'Spin_compound', 'Parity_compound', 'Deform_compound',
       'Decay_compound', 'Pairing_daughter', 'Shell_daughter', 'Spin_daughter',
       'Parity_daughter', 'Deform_daughter', 'Decay_daughter', 'ERG', 'XS', 'dXS',
		'unc_r', 'unc_hls', 'unc_sn', 'unc_sp', 'unc_ba', 'atomic_mass', 'unc_am', 'massexcess',
		'unc_me',]

ENDFBVIII_copy = pd.DataFrame(columns = merged_columns)
for i, row in tqdm.tqdm(ENDFBVIII_file.iterrows(), total=ENDFBVIII_file.shape[0]):
	target_nuclide = [row['Z'], row['A']]
	for j, data in livechart_data.iterrows():
		a = data['n'] + data['z']
		if [data['z'], a] == target_nuclide: # if match
			copy_row = row
			for idx, name in enumerate(new_cols):
				copy_row[name] = data[name]

			ENDFBVIII_copy = ENDFBVIII_copy._append(copy_row, ignore_index=True)
