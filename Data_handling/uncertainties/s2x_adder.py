import pandas as pd
import tqdm

data_file = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
s2x_file = pd.read_csv('S2x_csv_data.csv')

new_columns = ['S2N_U', 'S2P_U']




s2nu = []
s2pu = []
for i, row in tqdm.tqdm(data_file.iterrows(), total=data_file.shape[0]):
	Aval = row['A']
	Zval = row['Z']

	s2df = s2x_file[s2x_file.Z == Zval]
	s2df = s2df[s2df.A == Aval]

	s2nu.append(s2df['S2N_U'].values[0])
	s2pu.append(s2df['S2P_U'].values[0])


# ENDFBVIII_copy = pd.DataFrame(columns = merged_columns)
# for i, row in ENDFBVIII_file.iterrows():
# 	print(f"Row {i}/{len(ENDFBVIII_file)}")
# 	target_nuclide = [row['Z'], row['A']]
# 	for j, data in s2x_file.iterrows():
# 		if [data['Z'], data['A']] == target_nuclide:
# 			copy_row = row
# 			for idx, name in enumerate(new_columns):
# 				copy_row[name] = data[name]
#
# 			ENDFBVIII_copy = ENDFBVIII_copy._append(copy_row, ignore_index=True)
