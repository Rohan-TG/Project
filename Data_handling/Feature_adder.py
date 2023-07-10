import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
import shap
from sklearn.metrics import mean_squared_error, r2_score
import periodictable


df = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv') # new interpolated dataset, used for training only

df = df.drop(axis=1, labels=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0'])

Z_l_unpack, A_l_unpack, Nlow_unpack, Ulow_unpack, Ntop_unpack, Utop_unpack, ainf_unpack, = np.loadtxt("level-densities-ctmeff.txt",
													 usecols=(0,1,7,8,9,10,13),
													 skiprows=1,
													 unpack=True)


# unpacked_df = pd.read_csv("level-densities-ctmeff.txt",)

# unpacked_df = unpacked_df.dropna(axis=1,)

# print(unpacked_df)


unpacked_nuclides = []
for z, a in zip(Z_l_unpack, A_l_unpack):
	if [z,a] in unpacked_nuclides:
		continue
	else:
		unpacked_nuclides.append([z,a])


Nlow = []
Ulow = []
Ntop = []
Utop = []
ainf = []

for z, a in zip(df['Z'], df['A']): # nuclides in dataframe
	if [z,a] in unpacked_nuclides:

		unpack_nuc_index = unpacked_nuclides.index([z,a]) # fetch index of unpacked nuclide and add its corresponding features
		Nlow.append(Nlow_unpack[unpack_nuc_index])
		Ulow.append(Ulow_unpack[unpack_nuc_index])
		Ntop.append(Ntop_unpack[unpack_nuc_index])
		Utop.append(Utop_unpack[unpack_nuc_index])
		ainf.append(ainf_unpack[unpack_nuc_index])
	else:
		Nlow.append(np.nan)
		Ulow.append(np.nan)
		Ntop.append(np.nan)
		Utop.append(np.nan)
		ainf.append(np.nan)


df_new = df

df_new.insert(79, "Nlow", Nlow)
df_new.insert(80, "Ulow", Ulow)
df_new.insert(81, "Ntop", Ntop)
df_new.insert(82, "Utop", Utop)
df_new.insert(83, "ainf", ainf)
#
#
# print(df_new.columns)

# df_new.to_csv("level_densities_v1_zeroed_1_xs_fund_feateng.csv")