import pandas as pd
from scipy.interpolate import interp1d
# import the XS and ERG data only for MT16, MT103, MT107
from matrix_functions import range_setter, General_plotter
import matplotlib.pyplot as plt
import periodictable
import time
import numpy as np

ENDFBVIII_MT16 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
ENDFBVIII_MT103 = pd.read_csv('ENDFBVIII_MT103_fund_features_only.csv')
ENDFBVIII_MT107 = pd.read_csv('ENDFBVIII_MT107_fund_features_only.csv')

ENDF_B_nuclides = range_setter(df=ENDFBVIII_MT16, la=0,ua=260)
nucs103 = range_setter(df=ENDFBVIII_MT103, la=0,ua=260)
nucs107 = range_setter(df=ENDFBVIII_MT107, la=0,ua=260)

count = 0
# nucmass = [[],[],[]]
# avgxs = []
# maxxs = []

interpolated_MT103 = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])

for nuclide in ENDF_B_nuclides:
	if nuclide in nucs103:
		erg103, xs103 = General_plotter(df=ENDFBVIII_MT103, nuclides=[nuclide])
		erg16, xs16 = General_plotter(df=ENDFBVIII_MT16, nuclides=[nuclide])

		# nucmass[0].append(nuclide[1])
		# nucmass[1].append(max(xs103))
		# nucmass[2].append(np.mean(xs103))

		f103 = interp1d(x=erg103, y=xs103, fill_value='extrapolate')

		ixs103 = f103(erg16)

		# plt.figure()
		# plt.grid()
		# plt.plot(erg103, xs103, label = 'original')
		# plt.plot(erg16, ixs103, label = 'interpolated')
		# plt.legend()
		# plt.xlabel('Energy / MeV')
		# plt.ylabel('$\sigma_{n,p}$ / b')
		# plt.title(f"(n,p) cross sections for {periodictable.elements[nuclide[0]]}-{nuclide[1]}")
		# plt.show()
		# time.sleep(1)
		# count +=1
		# if count > 20:
		# 	break

# logxs = [np.log(i) for i in nucmass[2]]
# plt.figure()
# # plt.plot(nucmass[0], nucmass[1], label = 'max xs')
# plt.plot(nucmass[0], logxs, 'x', label = 'mean xs')
# plt.xlabel('A')
# plt.legend()
# plt.grid()
# plt.ylabel('XS / b')
# plt.show()
