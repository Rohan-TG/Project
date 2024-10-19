import pandas as pd
from matrix_functions import range_setter
import tqdm
import numpy as np

wrongFile = pd.read_csv('ENDFBVIII_MT107_all_features_old.csv')
cols = wrongFile.columns
target_nuclides = range_setter(df=wrongFile, la=0, ua=300)

fund = pd.read_csv(r'/mnt/c/Users/TG300/Project/Data_handling/1_fund.csv')
fundnucs = range_setter(df=fund, la=0, ua=300)

# dock = pd.read_csv('CENDL-3.2_MT107_all_features_old.csv')


Sn_daughter_list = []
S2n_daughter_list = []
Sp_daughter_list = []
S2p_daughter_list = []
Radius_daughter_list = []
ME_daughter_list = []
BEA_daughter_list = []
BEA_A_daughter_list = []
Pairing_daughter_list = []
Shell_daughter_list = []
Spin_daughter_list = []
Parity_daughter_list = []
Deform_daughter_list = []
Decay_daughter_list = []
Asymmetry_daughter_list = []



for n in tqdm.tqdm(target_nuclides, total=len(target_nuclides)):
	currentnuc = [n[0], n[1]]
	alpha_daughter_Z = currentnuc[0] - 2
	alpha_daughter_A = currentnuc[1] - 3
	alpha_daughter_nuclide = [alpha_daughter_Z, alpha_daughter_A]

	alpha_daughter_N = alpha_daughter_A - alpha_daughter_Z
	if alpha_daughter_A > 0:
		daughter_asymmetry = (alpha_daughter_N - alpha_daughter_Z) / alpha_daughter_A
	else:
		daughter_asymmetry = np.nan

	reduced_df = wrongFile[(wrongFile['Z'] == currentnuc[0]) & (wrongFile['A'] == currentnuc[1])]

	if alpha_daughter_nuclide in fundnucs:
		for j, feature_row in fund.iterrows():
			if [feature_row['Z'], feature_row['A']] == alpha_daughter_nuclide:
				for i, row in reduced_df.iterrows():
					# dummy_row = feature_row
					#
					# dummy_row['XS'] = row['XS']
					# dummy_row['ERG'] = row['ERG']
					#
					# fixed_df = fixed_df._append(dummy_row, ignore_index=True)


					Sn_daughter_list.append(feature_row['Sn_daughter'])
					S2n_daughter_list.append(feature_row['S2n_daughter'])
					Sp_daughter_list.append(feature_row['Sp_daughter'])
					S2p_daughter_list.append(feature_row['S2p_daughter'])
					Radius_daughter_list.append(feature_row['Radius_daughter'])
					ME_daughter_list.append(feature_row['ME_daughter'])
					BEA_daughter_list.append(feature_row['BEA_daughter'])
					BEA_A_daughter_list.append(feature_row['BEA_A_daughter'])
					Pairing_daughter_list.append(feature_row['Pairing_daughter'])
					Shell_daughter_list.append(feature_row['Shell_daughter'])
					Spin_daughter_list.append(feature_row['Spin_daughter'])
					Parity_daughter_list.append(feature_row['Parity_daughter'])
					Deform_daughter_list.append(feature_row['Deform_daughter'])
					Decay_daughter_list.append(feature_row['Decay_daughter'])
					Asymmetry_daughter_list.append(daughter_asymmetry)
	else:
		for idx, rowx in reduced_df.iterrows():
			Sn_daughter_list.append(np.nan)
			S2n_daughter_list.append(np.nan)
			Sp_daughter_list.append(np.nan)
			S2p_daughter_list.append(np.nan)
			Radius_daughter_list.append(np.nan)
			ME_daughter_list.append(np.nan)
			BEA_daughter_list.append(np.nan)
			BEA_A_daughter_list.append(np.nan)
			Pairing_daughter_list.append(np.nan)
			Shell_daughter_list.append(np.nan)
			Spin_daughter_list.append(np.nan)
			Parity_daughter_list.append(np.nan)
			Deform_daughter_list.append(np.nan)
			Decay_daughter_list.append(np.nan)
			Asymmetry_daughter_list.append(np.nan)


wrongFile['Sn_daughter'] = Sn_daughter_list
wrongFile['S2n_daughter'] = S2n_daughter_list
wrongFile['Sp_daughter'] = Sp_daughter_list
wrongFile['S2p_daughter'] = S2p_daughter_list
wrongFile['Radius_daughter'] = Radius_daughter_list
wrongFile['ME_daughter'] = ME_daughter_list
wrongFile['BEA_daughter'] = BEA_daughter_list
wrongFile['BEA_A_daughter'] = BEA_A_daughter_list
wrongFile['Pairing_daughter'] = Pairing_daughter_list
wrongFile['Shell_daughter'] = Shell_daughter_list
wrongFile['Spin_daughter'] = Spin_daughter_list
wrongFile['Parity_daughter'] = Parity_daughter_list
wrongFile['Deform_daughter'] = Deform_daughter_list
wrongFile['Decay_daughter'] = Decay_daughter_list
wrongFile['Asymmetry_daughter'] = Asymmetry_daughter_list

wrongFile.to_csv('ENDFBVIII_MT_107_all_features.csv')