import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable

df = pd.read_csv("1_fund.csv")

df_for_features = pd.read_csv("zeroed_1_xs_fund_feateng.csv")

al = []
for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
	if [j, i] in al:
		continue
	else:
		al.append([j, i]) # format is [Z, A]
		# print([j,i])
df_for_features= df_for_features.drop(labels=['Unnamed: 0.2',
											  'Unnamed: 0.1', 'Unnamed: 0','XSlow',
											  'XSupp', 'MT', 'erg_001pc', 'xs_001pc',
											  'erg_50pc', 'xs_50pc','erg_90pc',
											  'xs_90pc', 'erg_max', 'xs_max',
											  'max_int', '50_pc_int', '0_1_pc_int', '90_pc_int'],
									  axis =1)
# print(df_for_features.columns)
# print(df.columns)
original_row = df_for_features.head(1)

# iteration_features =


all_new_df_features = df.columns

features_without_XS_ERG = df_for_features.drop(labels=['XS', 'ERG'], axis = 1)
useable_features = features_without_XS_ERG.columns







def UNPACKER(nuclides):

	"""nuclides argument must be a list of nuclides with elements in the format [z,a]
	returns two lists:

	XS_list: list of concatenated XS data
	ERG_list: as above"""

	# XS_list = []
	# ERG_list = []
	# z_list = []
	# a_list = []

	new_dataframe = pd.DataFrame(columns=useable_features)

	for nuc in nuclides:

		retrieval_element_string = periodictable.elements[nuc[0]]

		if nuc[1] < 10:
			A_string = "00" + str(nuc[1])
		elif nuc[1] > 10 and nuc[1] < 100:
			A_string = "0" + str(nuc[1])
		elif nuc[1] > 99:
			A_string = str(nuc[1])

		retrieval_string = f"n-{retrieval_element_string}{A_string}-MT016.txt"

		# print(retrieval_string)

		unpacked_ERG, unpacked_XS = np.loadtxt(retrieval_string, skiprows=5, unpack=True)

		unpacked_XS = [(xs/1000) for xs in unpacked_XS]

		# z_unpacked = [nuc[0] for i in range(len(unpacked_ERG))]
		# a_unpacked = [nuc[1] for i in range(len(unpacked_XS))]


		for i, row in df.iterrows():
			if [row['Z'], row['A']] == [nuc[0],nuc[1]]:
				dummy_row = original_row

				for feature in useable_features:
					row_feature = row[str(feature)]
					dummy_row[str(feature)] = row_feature

				for xs, energy in zip(unpacked_XS, unpacked_ERG):
					dummy_row['XS'] = xs
					dummy_row['ERG'] = energy


					# print(dummy_row)

					new_dataframe = pd.concat([new_dataframe, dummy_row])


		# XS_list = XS_list + unpacked_XS
		# ERG_list = ERG_list + unpacked_ERG
		# z_list = z_list + z_unpacked
		# a_list = a_list + a_unpacked


	# for i, row in df.iterrows():
	# 	for z, a in zip(z_list, a_list):
	# 		if [row['Z'], row['A']] == [z,a]:
	# 			dummy_row = original_row
	#
	# 			# temp_df = pd.DataFrame(columns=useable_features)
	# 			for feature in useable_features:
	# 				dummy_row[feature] = row[feature]
	#
	# 			for j, (xs, energy) in enumerate(zip(XS_list, ERG_list)):
	# 				if z_list[j] == z and a_list[j] == a:
	#
	# 					dummy_row['XS'] = xs
	# 					dummy_row['ERG'] = energy
	#
	# 				new_dataframe._append(dummy_row, ignore_index=True)


	return new_dataframe



# list of all the nuclides' XSs being added. The format as usual is [z,a]
new_nuclides = [[6,12],
				[8,16],
				[3,7],
				[94,239],
				[95,240],
				[89,225],
				[89,226],
				[89,227],
				[95,240],
				[95,241],
				[95,242],
				[95,243],
				[95,244],
				[5,11],
				[4,9],
				[83,209],
				[97,245],
				[97,246],
				[97,247],
				[97,248],
				[97,250],
				[98,246],
				[98,248],
				[98,249],
				[98,250],
				[98,251],
				[98,252],
				[98,253],
				[96,240],
				[96,241],
				[96,243],
				[96,244],
				[96,246],
				[96,247],
				[96,248],
				[99,252],
				[99,253],
				[99,254],
				[99,255],
				[9,19],
				[100,255],
				[1,2],
				[1,3],
				[3,7],
				[12,24],
				[12,25],
				[12,26],
				[7,14],
				[7,15],
				[11,22],
				[10,20],
				[10,21],
				[10,22],
				[93,234],
				[93,235],
				[93,237],
				[93,238],
				[93,239],
				[8,16],
				[8,17],
				[8,18],
				[91,229],
				[91,230],
				[91,231],
				[91,232],
				[84,208],
				[84,209],
				[84,210],
				[94,236],
				[94,237],
				[94,238],
				
				]

WIP_df = UNPACKER(new_nuclides)

print(WIP_df)

plt.plot(WIP_df['ERG'], WIP_df['XS'])
plt.show()