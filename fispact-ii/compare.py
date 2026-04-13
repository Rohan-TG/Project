import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# name = 'Pr-141'
# exp_filename = 'Pr_data/2000exp_5min.exp'

name = 'Ga-69'
exp_filename = 'Ga_data/2000exp_5min.exp'

exp = np.loadtxt(exp_filename, delimiter=' ')
exp = exp.transpose()
exp_minutes = exp[0]
exp_decay_heat = exp[1]
exp_decay_heat_uncert = exp[2]


# Following for Pr-141 (n,2n) only
# ML_values = np.array([ #9.48359e-06,
# 						  5.95854e-06, 5.57926e-06, 5.28812e-06, 5.00435e-06,
#     4.75428e-06, 4.33610e-06, 3.83542e-06, 3.39257e-06, 2.83196e-06,
#     2.26156e-06, 1.80606e-06, 1.28891e-06, 8.38998e-07, 5.44294e-07,
#     2.87842e-07, 1.24115e-07, 5.35554e-08, 2.31462e-08, 5.52851e-09,
#     8.04572e-10, 2.05935e-10])



# ML_uncert = [
#    # 2.46534e-07,
# 	2.18065e-07, 2.06493e-07, 1.96203e-07, 1.85791e-07,
#     1.76533e-07, 1.61015e-07, 1.42425e-07, 1.25981e-07, 1.05164e-07,
#     8.39824e-08, 6.70670e-08, 4.78618e-08, 3.11538e-08, 2.02093e-08,
#     1.06854e-08, 4.60503e-09, 1.98463e-09, 8.55301e-10, 2.01058e-10,
#     2.62288e-11, 6.98387e-12
# ]

#############################

# TENDL_values = np.array([
#     # 9.50562e-06,
# 	5.97689e-06, 5.59662e-06, 5.30462e-06, 5.01995e-06,
#     4.76911e-06, 4.34963e-06, 3.84739e-06, 3.40316e-06, 2.84080e-06,
#     2.26861e-06, 1.81169e-06, 1.29293e-06, 8.41614e-07, 5.45989e-07,
#     2.88740e-07, 1.24501e-07, 5.37219e-08, 2.32190e-08, 5.54549e-09,
#     8.06353e-10, 2.06046e-10
# ])

# TENDL_uncert = np.array([
#     # 2.47303e-07,
# 	2.18745e-07, 2.07137e-07, 1.96815e-07, 1.86371e-07,
#     1.77084e-07, 1.61517e-07, 1.42869e-07, 1.26375e-07, 1.05492e-07,
#     8.42443e-08, 6.72761e-08, 4.80111e-08, 3.12509e-08, 2.02723e-08,
#     1.07188e-08, 4.61939e-09, 1.99081e-09, 8.58004e-10, 2.01688e-10,
#     2.62931e-11, 6.98580e-12
# ])


#####################
# Following for Ga-69

ML_values = np.array([    #5.30708e-06,
						  1.65133e-06, 1.15531e-06, 1.04167e-06, 1.00361e-06,
    9.86574e-07, 9.73666e-07, 9.56257e-07, 9.35333e-07, 9.15970e-07,
    8.91176e-07, 8.62592e-07, 8.36475e-07, 8.01283e-07, 7.60993e-07,
    7.24752e-07, 6.77446e-07, 6.23217e-07, 5.76009e-07, 5.34240e-07,
    4.72047e-07, 4.00236e-07, 3.43252e-07])

ML_uncert = np.array([    #6.62717e-08,
						  6.37887e-08, 6.24196e-08, 6.13936e-08, 6.04659e-08,
    5.96971e-08, 5.89509e-08, 5.78512e-08, 5.64893e-08, 5.52302e-08,
    5.36424e-08, 5.18620e-08, 5.02829e-08, 4.82157e-08, 4.59114e-08,
    4.38750e-08, 4.12444e-08, 3.82461e-08, 3.56445e-08, 3.33513e-08,
    2.99577e-08, 2.60643e-08, 2.29621e-08])

##########################################

TENDL_values = np.array([    #5.30105e-06,
					1.64356e-06, 1.14739e-06, 1.03376e-06, 9.95686e-07,
    9.78694e-07, 9.65819e-07, 9.48443e-07, 9.27565e-07, 9.08252e-07,
    8.83527e-07, 8.55029e-07, 8.28996e-07, 7.93928e-07, 7.53795e-07,
    7.17708e-07, 6.70624e-07, 6.16676e-07, 5.69738e-07, 5.28229e-07,
    4.66459e-07, 3.95197e-07, 3.38707e-07])

TENDL_uncert = np.array([   # 6.58961e-08,
					6.34011e-08, 6.20256e-08, 6.09951e-08, 6.00634e-08,
    5.92915e-08, 5.85423e-08, 5.74386e-08, 5.60718e-08, 5.48085e-08,
    5.32158e-08, 5.14305e-08, 4.98477e-08, 4.77767e-08, 4.54697e-08,
    4.34323e-08, 4.08025e-08, 3.78078e-08, 3.52122e-08, 3.29265e-08,
    2.95487e-08, 2.56821e-08, 2.26089e-08])


ML_values = ML_values * 1e6
ML_uncert = np.array(ML_uncert) * 1e6

TENDL_values = TENDL_values * 1e6


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