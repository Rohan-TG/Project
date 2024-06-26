import pandas as pd
import tqdm

from matrix_functions import range_setter
import numpy as np


library = pd.read_csv('part1zeroed.csv')
print(library['ERG'])

library['ERG'] = library['ERG']
print(library['ERG'])

nuclide_feature_df = pd.read_csv('1_fund.csv')

library_nuclides = range_setter(df=library, la=0, ua=300)


nuclides_for_extraction = library_nuclides

print(len(nuclides_for_extraction))
print(len(library_nuclides))

columns = list(nuclide_feature_df.columns)

library = library[library.ERG < 20.0]
# library = library[library.LISO == 0]
library.index = range(len(library))


# target_nuclides = nuclides_for_extraction
target_nuclides = nuclides_for_extraction
nuclide_feature_df.index = range(len(nuclide_feature_df))

def feature_engineer(df):

	df_r = pd.DataFrame(columns=columns)

	for lx, target_nuclide in enumerate(tqdm.tqdm(target_nuclides, total=len(target_nuclides))):
		# print(f"{lx+1}/{len(target_nuclides)}")
		for j, feature_row in nuclide_feature_df.iterrows():
			if [feature_row['Z'], feature_row['A']] == target_nuclide:
				for i, row in df.iterrows():
					if [row['Z'], row['A']] == target_nuclide:
						dummy_row = feature_row
						dummy_row['XS'] = row['XS']
						dummy_row['ERG'] = row['ERG']
						dummy_row['dXS'] = row['dXS']

						df_r = df_r._append(dummy_row, ignore_index=True)

	return df_r



library_with_features = feature_engineer(df=library)
# library_with_features.to_csv("NEW_TENDL21_correctly_featured_1_fund_only.csv")
print(library_with_features.shape)
