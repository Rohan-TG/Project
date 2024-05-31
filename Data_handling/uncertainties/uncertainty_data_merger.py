import pandas as pd
import tqdm
import numpy as np

livechart_data = pd.read_csv('livechart_ground_state_data.csv')
library_file = pd.read_csv('TENDL_2021_MT16_S2x_uncertainties.csv')


library_file = library_file.drop(columns=['Unnamed: 0'])



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
       'Parity_daughter', 'Deform_daughter', 'Decay_daughter', 'ERG', 'XS',
       'Nlow', 'Ulow', 'Ntop', 'Utop', 'ainf', 'XSupp', 'XSlow', 'Asymmetry',
       'Asymmetry_compound', 'Asymmetry_daughter',
		]

new_cols = ['unc_r', 'unc_hls', 'unc_sn', 'unc_sp', 'unc_ba', 'atomic_mass', 'unc_am', 'massexcess',
			'unc_me']

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
       'Parity_daughter', 'Deform_daughter', 'Decay_daughter', 'ERG', 'XS',
       'Nlow', 'Ulow', 'Ntop', 'Utop', 'ainf', 'XSupp', 'XSlow', 'Asymmetry',
       'Asymmetry_compound', 'Asymmetry_daughter',
		'unc_r', 'unc_hls', 'unc_sn', 'unc_sp', 'unc_ba', 'atomic_mass', 'unc_am', 'massexcess',
		'unc_me', 'half_life_sec']


uncr = []
unchls = []
hls_seconds = []
uncsn = []
uncsp = []
uncba = []
unc_atomicmass = []
atomicmass = []
uncme = []


livechart_nuclides = []
for i, row in livechart_data.iterrows():
	zval = row['z']
	nval = row['n']

	aval = zval + nval
	if [zval, aval] not in livechart_nuclides:
		livechart_nuclides.append([zval,aval])

for i, row in tqdm.tqdm(library_file.iterrows(), total = library_file.shape[0]):
	Aval = row['A']
	Zval = row['Z']


	df = livechart_data[livechart_data.z == Zval]
	Nval = Aval - Zval
	df = df[df.n == Nval]

	try:

		uncr.append(df['unc_r'].values[0])
		unchls.append(df['unc_hls'].values[0])
		hls_seconds.append(df['half_life_sec'].values[0])
		uncsn.append(df['unc_sn'].values[0])
		uncsp.append(df['unc_sp'].values[0])
		uncba.append(df['unc_ba'].values[0])
		atomicmass.append(df['atomic_mass'].values[0])
		uncme.append(df['unc_me'].values[0])
	except IndexError:
		uncr.append(np.nan)
		unchls.append(np.nan)
		hls_seconds.append(np.nan)
		uncsn.append(np.nan)
		uncsp.append(np.nan)
		uncba.append(np.nan)
		atomicmass.append(np.nan)
		uncme.append(np.nan)

print(library_file.shape)




# 74