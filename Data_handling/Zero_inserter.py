import time
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import periodictable


# df = pd.read_csv('interpolated_n2_1_xs_fund_feateng.csv') # new interpolated dataset, used for training only
df_test = pd.read_csv('TENDL21_features_truly_unzeroed.csv') # original dataset, used for validation



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

df_zero.to_csv('TENDL21_features_arange_zeroed.csv')

# to_reduce = pd.read_csv('TENDL21_MT16_XS.csv')
#
# to_reduce = to_reduce[to_reduce.ERG <20.0]
# to_reduce.index = range(len(to_reduce))
#
#
# to_reduce.to_csv('JENDL4_correctly_zeroed_with_features.csv')
# df_zero = pd.read_csv("Third_attempt_TENDL.csv")
# #
# df_using = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
# #
# from matrix_functions import range_setter
# al = range_setter(df=df_using, la=0, ua=260)
#
# TENDL_nuclides = range_setter(df=df_zero, la=0, ua=300)
#
# for nuc in TENDL_nuclides:
#
# 	if nuc in al:
#
# 		X_test, y_test = make_test(nuclides=[nuc], df=df_zero)
#
# 		endfx, endfy = make_test(nuclides=[nuc], df=df_using)
#
# 		plt.figure()
# 		plt.plot(X_test[:,2], y_test, label='TENDL')
# 		plt.grid()
# 		plt.title(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
#
# 		plt.plot(endfx[:,2], endfy, 'x-', label='ENDF/B')
# 		plt.legend()
# 		plt.show()
#
# 		time.sleep(1.5)
