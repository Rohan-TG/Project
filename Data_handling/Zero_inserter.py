import time
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import periodictable

df_test = pd.read_csv('JENDL5_all_features_unzeroed.csv')



def zero_maker(df):
	"""input df: Dataframe to be zeroed

	Function inserts 0 XS values for energies below reaction threshold.

	Returns: New dataframe with the additional values."""

	df_zeroed = pd.DataFrame(columns=df.columns)  # empty dataframe with column names only, to be appended to
	current_nuclide = [] # iteration nuclide

	for i, row in df.iterrows():
		if i % 1000 == 0:
			print(f"{i}/{len(df['Z'])}")

		if [row['Z'], row['A']] != current_nuclide: # for new nuclide in iteration, insert n number of 0 values

			min_energy = row['ERG'] # Threshold energy
			# energy_app_list = np.linspace(0, min_energy-0.1, 20) # 20 values, 0 to threshold MeV - 0.1
			energy_app_list = np.arange(0.0, min_energy, 0.1)

			dummy_row = row # to be copied
			dummy_row['XS'] = 0.0 # set XS value to 0 for all energies below threshold

			for erg in energy_app_list:
				dummy_row['ERG'] = erg # set energy value
				df_zeroed = df_zeroed._append(dummy_row, ignore_index=True) # append new row

			current_nuclide = [row['Z'],row['A']] # once iteration complete, update the iteration nuclide
		else:
			df_zeroed = df_zeroed._append(row, ignore_index=True) # if below-threshold values have already been added, append normal row

		# print(current_nuclide) # tracker

	df_zeroed.index = range(len(df_zeroed)) # reindex dataframe

	return df_zeroed



df_zero = zero_maker(df=df_test)
print(df_zero.head())

df_zero.to_csv('JENDL5_arange_all_features.csv')

