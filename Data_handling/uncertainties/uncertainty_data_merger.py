import pandas as pd
import tqdm
import numpy as np

livechart_data = pd.read_csv('livechart_ground_state_data.csv')
library_file = pd.read_csv('TENDL_2021_MT16_S2x_uncertainties.csv')


library_file = library_file.drop(columns=['Unnamed: 0'])




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