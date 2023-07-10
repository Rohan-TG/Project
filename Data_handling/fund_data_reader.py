import pandas as pd
import numpy as np
from matrix_functions import range_setter

df = pd.read_csv("1_xs_fund.csv")


df = df[df.MT == 16]
df.index = range(len(df))
print("\n MT=16 isolated")


def asymmetry_term(N, Z, A):

    s_target = (N - Z)/A

    compound_N = N+1
    compound_A = A+1
    s_compound = (compound_N - Z) / compound_A

    daughter_N = N-1
    daughter_A = A-1
    s_daughter = (daughter_N - Z)/daughter_A

    return s_target, s_compound, s_daughter

s, s_c, s_d = asymmetry_term(N=df['N'], Z=df['Z'], A=df['A'])

df.insert(64, 'Asymmetry', value=s)
df.insert(65, 'Asymmetry_compound', value=s_c)
df.insert(66, 'Asymmetry_daughter', value=s_d)


Z_l_unpack, A_l_unpack, Nlow_unpack, Ulow_unpack, Ntop_unpack, Utop_unpack, ainf_unpack, = np.loadtxt("level-densities-ctmeff.txt",
													 usecols=(0,1,7,8,9,10,13),
													 skiprows=1,
													 unpack=True)

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

df_new.insert(67, "Nlow", Nlow)
df_new.insert(68, "Ulow", Ulow)
df_new.insert(69, "Ntop", Ntop)
df_new.insert(70, "Utop", Utop)
df_new.insert(71, "ainf", ainf)


print(len(df_new.columns))
print(df_new.columns)

# df.to_csv("Nuclear_Data_features.csv")