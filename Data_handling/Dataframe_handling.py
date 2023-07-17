import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy
import scipy.stats
from matrix_functions import log_make_test, range_setter

df = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')


df_corr = df[df.A >= 30]
mid_range = df_corr[df_corr.A <= 212]

mid_range = df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1',
							 'Unnamed: 0','XSupp', 'XSlow',
							 'Z_even', 'A_even', 'N_even',
							 'Sp_daughter','Parity_daughter',
							 'S2n_compound','Shell_daughter',
							 'S2p_daughter','ME_daughter','ainf',
							 'Asymmetry_compound','Ulow', 'Ntop',
							 'Spin_daughter', 'Parity_compound',
							 'Radius_daughter','Pairing_daughter',
							 'Decay_compound','octopole_deformation',
							 'Deform','Pairing_compound'])




mid_range.index = range(len(mid_range))

correlations = mid_range.corr()



columns = mid_range.columns



print(columns)

plt.figure(figsize=(50,50))
plt.matshow(correlations)
plt.xticks(np.arange(len(columns)), columns, fontsize=5, rotation=90)
plt.yticks(np.arange(len(columns)), columns, fontsize=5)
cb = plt.colorbar()
plt.show()