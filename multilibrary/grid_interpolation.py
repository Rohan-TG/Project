import pandas as pd
import numpy as np
from matrix_functions import General_plotter, range_setter
import scipy

cendl = pd.read_csv('CENDL32_all_features.csv')
jendl = pd.read_csv('JENDL5_arange_all_features.csv')
jeff = pd.read_csv('JEFF33_all_features.csv')
tendl = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
endfb = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')


grid = np.arange(0.0, 20.0, 0.1)

# master_columns = ['Z', 'A', 'ERG', 'XS']

def df_interpolator(df, energy_grid=grid):
	"""Takes original dataframe, interpolates XS onto desired energy grid, then remakes
	dataframe with all the original features included in one go."""

	columns = df.columns
	df_nuclides = range_setter(df=df, la=0, ua=260)

	full_df = pd.DataFrame(columns=columns)

	for n, nuclide in enumerate(df_nuclides):
		print(f"{n+1}/{len(df_nuclides)} Nuclides")
		nuclide_energy, nuclide_xs = General_plotter(df=df, nuclides=[nuclide])
		i_function = scipy.interpolate.interp1d(x=nuclide_energy, y=nuclide_xs, fill_value='extrapolate')

		new_xs = i_function(energy_grid)
		nuc_zlist = [nuclide[0] for idx in energy_grid]
		nuc_alist = [nuclide[1] for idx in energy_grid]

		temp_maindf = pd.DataFrame({'Z': nuc_zlist, 'A': nuc_alist, 'ERG': energy_grid, 'XS': new_xs}) # define temporary new dataframe with new grid

		for i2, feature_row in df.iterrows():
			if [feature_row['Z'], feature_row['A']] == nuclide:
				features = feature_row
				break

		for i, row in temp_maindf.iterrows():

			dummy_row = features
			dummy_row['ERG'] = row['ERG']
			dummy_row['XS'] = row['XS']

			full_df = full_df._append(dummy_row, ignore_index=True)

	return full_df

trial = df_interpolator(df=endfb)




# def grid_interpolator(df):
# 	"""Takes original dataframe, interpolates the cross sections onto the new grid, then
# 	returns a new dataframe with the same columns, but on the new energy grid."""
#
# 	df_nuclides = range_setter(df=df, la=0, ua=260)
#
# 	for nuclide in df_nuclides:
# 		cross_sections =
# 	pass