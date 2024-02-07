import pandas as pd
import numpy as np

S2x_file =  open('/mnt/c/Users/TG300/Project/Data_handling/uncertainties/rct1mas20.txt')

lines = S2x_file.readlines()

Zlist = []
Alist = []
S2n_unc_list = []
S2p_unc_list = []

S2n_list = []
S2p_list = []

for i, row in enumerate(lines):
	if i > 35:
		z = int(row[8:14])
		a = int(row[1:5])

		S2nu = ''
		S2pu = ''

		temps2nu = row[25:36]
		temps2pu = row[47:58]

		if '*' in temps2nu:
			S2nu = np.nan
		if '*' in temps2pu:
			S2pu = np.nan

		print(temps2nu)




		# print(row[15:25])