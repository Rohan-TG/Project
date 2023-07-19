import pandas as pd
from matrix_functions import range_setter
import numpy as np


library = pd.read_csv('TENDL_MT16_XS_NOLISO_LFS.csv')
nuclide_feature_df = pd.read_csv('1_fund.csv')
ENDFBVIII = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')

library_nuclides = range_setter(df=library, la=0, ua=300)

nuclides_already_in_ENDFBVIII = range_setter(la=0, ua=300, df=ENDFBVIII)


nuclides_for_extraction = library_nuclides

print(len(nuclides_for_extraction))
print(len(nuclides_already_in_ENDFBVIII))
print(len(library_nuclides))

columns = list(nuclide_feature_df.columns)

library = library[library.ERG < 20]
library.index = range(len(library))

print(library.columns)
library = library.drop(columns=['Unnamed: 0'])
print(library.columns)
print(library.head())
target_nuclides = nuclides_for_extraction

nuclide_feature_df.index = range(len(nuclide_feature_df))

def feature_engineer(df):

	df_r = pd.DataFrame(columns=columns)

	for lx, target_nuclide in enumerate(target_nuclides):
		print(f"{lx+1}/{len(target_nuclides)}")
		for j, feature_row in nuclide_feature_df.iterrows():
			if [feature_row['Z'], feature_row['A']] == target_nuclide:
				for i, row in df.iterrows():
					if [row['Z'], row['A']] == target_nuclide:
						dummy_row = feature_row
						dummy_row['XS'] = row['XS']
						dummy_row['ERG'] = row['ERG']

						# df_r = pd.concat([df_r, dummy_row], ignore_index=True, axis=1)
						df_r = df_r._append(dummy_row, ignore_index=True)

	return df_r

# def TENDL_feature_engineer(df):
#
# 	df_r = pd.DataFrame(columns=columns)
# 	filled = 0
# 	for j, feature_row in nuclide_feature_df.iterrows():
# 		print(f"{j + 1}/3429")
# 		for i, row in df.iterrows():
# 			if [row['Z'], row['A']] == [feature_row['Z'], feature_row['A']]:
# 				dummy_row = feature_row
# 				dummy_row['XS'] = row['XS']
# 				dummy_row['ERG'] = row['ERG']
#
# 				df_r = df_r._append(dummy_row, ignore_index=True)
# 				filled = 1
# 			elif filled > 0:
# 				filled = 0
# 				break
# 		if j > 25:
# 			break
#
# 	return df_r


library_with_features = feature_engineer(df=library)
library_with_features.to_csv("TENDL21_features_truly_unzeroed.csv")
print(library_with_features.shape)