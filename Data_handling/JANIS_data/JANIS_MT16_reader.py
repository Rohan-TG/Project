import pandas as pd
import periodictable
import numpy as np
import matplotlib.pyplot as plt
from matrix_functions import General_plotter, range_setter, error_fetcher
import tqdm
import scipy.interpolate

ENDFBVIII = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')

df = pd.read_csv('MT16_error_data.csv',
				 # header=2,
				 delimiter=',',
				 )
cols = df.columns
cols = cols[1:]
cols_short = [i[0:6] for i in cols]


new_names = []
for name in cols_short:
	new_names.append(name.strip())


parity_names = []
final_headers = []
for name in new_names:
	if name in parity_names:
		delta_header = name + ' ds'
		final_headers.append(delta_header)
	else:
		final_headers.append(name)
		parity_names.append(name)

final_headers.insert(0, 'Incident energy')
df2 = df.set_axis(final_headers, axis=1)

def nuclide_extractor(head):
	element_name = ''
	isotope_string = ''
	for char in head:
		if char == ' ':
			break
		try:
			integer = int(char)
			isotope_string += char
		except ValueError:
			element_name += char

	isotope = int(isotope_string)
	atomic_number = periodictable.elements.symbol(element_name).number
	return(atomic_number, isotope)

for header in df2.columns[1:]:
	nuclide_extractor(head=header)

mt16std = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])

ERG = []
Z = []
A = []
XS = []
dXS = []
for i, row in tqdm.tqdm(df2.iterrows(), total=df2.shape[0]):
	# print(f"{i}/{len(df2)}")
	if i > 1:

		parity_bits = [1]

		for name in final_headers[1:]:
			if 'ds' not in name:
				z, a = nuclide_extractor(head=name)
				Z.append(z)
				A.append(a)


				row_energy = row['Incident energy']

				if type(row_energy) == str:
					energy = float(row_energy.strip()) / 1e6
					ERG.append(energy)
				elif type(row_energy) == float:
					# print(row_energy)
					ERG.append(row_energy / 1e6)
				elif type(row_energy) == int:
					ERG.append(row_energy / 1e6)

				if parity_bits[-1] == 0:
					dXS.append(np.nan)

				if type(row[name]) == str and (row[name].strip()) != '':
					xs_value = float(row[name].strip())
					XS.append(xs_value)
				elif type(row[name]) == float:
					XS.append(row[name])
				elif row[name].strip() == '':
					XS.append(np.nan)
				parity_bits.append(0)
			if 'ds' in name:
				parity_bits.append(1)
				if type(row[name]) == str and (row[name].strip()) != '':
					dxs_value = float(row[name].strip())
					dXS.append(dxs_value)
				elif type(row[name]) == float:
					dXS.append(row[name])
				elif row[name].strip() == '':
					dXS.append(np.nan)

temp = pd.DataFrame({'Z': Z,
					 'A': A,
					 'ERG': ERG,
					 'XS': XS,
					 'dXS': dXS})

temptarget = [62,151]
erg, xs = General_plotter(df=temp, nuclides=[temptarget])
# logerg = [np.log10(i) for i in erg]
# logxs = [np.log10(i) for i in xs]

plt.figure()
plt.plot(erg, xs)
plt.xlabel('Energy')
plt.ylabel('b')
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[temptarget[0]]}-{temptarget[1]}")
plt.grid()
plt.show()


error_nuclides = range_setter(df=temp, la=0, ua=290)








def zero_maker(df):
	"""input df: Dataframe to be zeroed

	Function inserts 0 XS values for energies below reaction threshold.

	Returns: New dataframe with the additional values."""

	df_zeroed = pd.DataFrame(columns=df.columns)  # empty dataframe with column names only, to be appended to
	current_nuclide = [] # iteration nuclide

	for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
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



# df_zero = zero_maker(df=temp)
# print(df_zero.head())