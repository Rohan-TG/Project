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
		Zlist.append(z)

		a = int(row[1:5])
		Alist.append(a)



		S2nu = ''
		S2nstring = ''

		S2pu = ''
		S2pstring = ''

		temps2nu = row[25:36]
		temps2n = row[14:25]

		temps2pu = row[46:56]
		temps2p = row[36:47]



		if '*' in temps2p:
			S2p_unc_list.append(np.nan)
			S2p_list.append(np.nan)
			# print('bomboclat')
		else:
			for p in temps2p:
				if p != '#':
					S2pstring += p
			print(S2pstring)
			S2p_list.append(float(S2pstring))

			try:
				for x in temps2pu:
					if x != '#':
						S2pu += x
				S2p_unc_list.append(float(S2pu))
			except ValueError:
				# print(f'Skipped line: {i, temps2nu}')
				print('')



		if '*' in temps2n:
			S2n_unc_list.append(np.nan)
			S2n_list.append(np.nan)
			# print('bomboclat')
		else:
			for p in temps2n:
				if p != '#':
					S2nstring += p
			print(S2nstring)
			S2n_list.append(float(S2nstring))

			try:
				for x in temps2nu:
					if x != '#':
						S2nu += x
				S2n_unc_list.append(float(S2nu))
			except ValueError:
				# print(f'Skipped line: {i, temps2nu}')
				print('')


print(S2n_unc_list)

new_data = {
	'Z': Zlist,
	'A': Alist,
	'S2N': S2n_list,
	'S2N_U': S2n_unc_list,
	'S2P': S2p_list,
	'S2P_U': S2p_unc_list
}

df = pd.DataFrame(new_data)