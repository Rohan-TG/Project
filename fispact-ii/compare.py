import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

exp_filename = 'Pr_data/2000exp_5min.exp'
exp = np.loadtxt(exp_filename, delimiter=' ')
exp = exp.transpose()
exp_minutes = exp[0]
exp_decay_heat = exp[1]
exp_decay_heat_uncert = exp[2]



ML_values = np.array([ #9.48359e-06,
						  5.95854e-06, 5.57926e-06, 5.28812e-06, 5.00435e-06,
    4.75428e-06, 4.33610e-06, 3.83542e-06, 3.39257e-06, 2.83196e-06,
    2.26156e-06, 1.80606e-06, 1.28891e-06, 8.38998e-07, 5.44294e-07,
    2.87842e-07, 1.24115e-07, 5.35554e-08, 2.31462e-08, 5.52851e-09,
    8.04572e-10, 2.05935e-10])

ML_values = ML_values * 1e6

ML_uncert = [
   # 2.46534e-07,
	2.18065e-07, 2.06493e-07, 1.96203e-07, 1.85791e-07,
    1.76533e-07, 1.61015e-07, 1.42425e-07, 1.25981e-07, 1.05164e-07,
    8.39824e-08, 6.70670e-08, 4.78618e-08, 3.11538e-08, 2.02093e-08,
    1.06854e-08, 4.60503e-09, 1.98463e-09, 8.55301e-10, 2.01058e-10,
    2.62288e-11, 6.98387e-12
]

ML_uncert = np.array(ML_uncert) * 1e6
#############################

TENDL_values = np.array([
    # 9.50562e-06,
	5.97689e-06, 5.59662e-06, 5.30462e-06, 5.01995e-06,
    4.76911e-06, 4.34963e-06, 3.84739e-06, 3.40316e-06, 2.84080e-06,
    2.26861e-06, 1.81169e-06, 1.29293e-06, 8.41614e-07, 5.45989e-07,
    2.88740e-07, 1.24501e-07, 5.37219e-08, 2.32190e-08, 5.54549e-09,
    8.06353e-10, 2.06046e-10
])

TENDL_values = TENDL_values * 1e6

TENDL_uncert = np.array([
    # 2.47303e-07,
	2.18745e-07, 2.07137e-07, 1.96815e-07, 1.86371e-07,
    1.77084e-07, 1.61517e-07, 1.42869e-07, 1.26375e-07, 1.05492e-07,
    8.42443e-08, 6.72761e-08, 4.80111e-08, 3.12509e-08, 2.02723e-08,
    1.07188e-08, 4.61939e-09, 1.99081e-09, 8.58004e-10, 2.01688e-10,
    2.62931e-11, 6.98580e-12
])
TENDL_uncert = TENDL_uncert * 1e6

plt.figure()
plt.plot(exp_minutes, exp_decay_heat, 'x-', label = 'Experiment')
plt.errorbar(exp_minutes, ML_values, yerr=ML_uncert, fmt='x-', capsize=2, label='ML')
plt.errorbar(exp_minutes, TENDL_values, yerr=TENDL_uncert, fmt='x-', label='TENDL')
plt.xscale('log')
plt.legend()
plt.grid()
# plt.yscale('log')
plt.show()

terrs = []
mlerrs = []
for tendl, ml, true in zip(TENDL_values, ML_values, exp_decay_heat):
	terr = ((tendl - true) / true ) * 100
	terrs.append(terr)

	mlerr = ((ml - true) / true ) * 100
	mlerrs.append(mlerr)

plt.figure()
plt.plot(exp_minutes, terrs, 'x-', label = 'TENDL error')
plt.plot(exp_minutes, mlerrs, 'x-', label = 'ML error')
plt.grid()
plt.legend()
plt.show()