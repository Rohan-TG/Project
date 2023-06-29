import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# c12_endfb8_ERG, c12_endfb8_XS = np.loadtxt('n-C012-MT016.endfb8.0.txt', skiprows=5, unpack=True)


u238_endfb8_ERG, u238_endfb8_XS = np.loadtxt('n-Pu239-MT016.txt', skiprows=5, unpack=True, usecols=(0,1))


# plt.plot(c12_endfb8_ERG, c12_endfb8_XS, label = 'XS')
# plt.grid()
# plt.title("(n,2n) XS for C-12")
# plt.xlabel("Energy / MeV")
# plt.ylabel("XS / b")
# plt.show()


plt.plot(u238_endfb8_ERG, u238_endfb8_XS, label = 'XS')
plt.grid()
plt.title("(n,2n) XS for U-238")
plt.xlabel("Energy / MeV")
plt.ylabel("XS / b")
plt.show()




def dataframe_maker(nuclide_Z_list, nuclide_A_list, nuclide_XS, nuclide_ERG, Sn, S2n, Sp, S2p):

	Z_list = []
	A_list = []
	XS_list = []
	ERG_list = []
	Sn_list = []
	S2n_list = []
	Sp_list = []
	S2p_list = []

	for xs in nuclide_XS:
		pass

	pass