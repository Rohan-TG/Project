import pandas as pd
from scipy.interpolate import interp1d
# import the XS and ERG data only for MT16, MT103, MT107
from matrix_functions import range_setter, General_plotter
import matplotlib.pyplot as plt
import periodictable
import time
import numpy as np

ENDFBVIII_MT16 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
ENDFBVIII_MT16 = ENDFBVIII_MT16[ENDFBVIII_MT16.ERG <= 20.0]
ENDFBVIII_MT16.index = range(len(ENDFBVIII_MT16))
ENDFBVIII_MT103 = pd.read_csv('ENDFBVIII_MT103_fund_features_only.csv')
ENDFBVIII_MT107 = pd.read_csv('ENDFBVIII_MT107_fund_features_only.csv')


patient_library = pd.read_csv('CENDL32_all_features.csv')
patient_nuclides = range_setter(df=patient_library, la=0, ua=260)

ENDF_B_nuclides = range_setter(df=ENDFBVIII_MT16, la=0,ua=260)
nucs103 = range_setter(df=ENDFBVIII_MT103, la=0,ua=260)
nucs107 = range_setter(df=ENDFBVIII_MT107, la=0,ua=260)

count = 0
# nucmass = [[],[],[]]
# avgxs = []
# maxxs = []

interpolated_MT103 = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])



patient_library['MT103XS'] = pd.Series(index=patient_library.index)
patient_library['MT107XS'] = pd.Series(index=patient_library.index)

main_columns = patient_library.columns
new_df = pd.DataFrame(columns=main_columns)

MT103_df = pd.DataFrame(columns=['Z', 'A', 'MT103XS', 'ERG'])


new_df.index = range(len(new_df))



for num, nuclide in enumerate(patient_nuclides):
	print(f"{num}/{len(patient_nuclides)}")
	if nuclide in nucs103 and nuclide in nucs107:

		erg103, xs103 = General_plotter(df=ENDFBVIII_MT103, nuclides=[nuclide])
		erg16, xs16 = General_plotter(df=patient_library, nuclides=[nuclide])
		erg107, xs107 = General_plotter(df=ENDFBVIII_MT107, nuclides=[nuclide])

		f103 = interp1d(x=erg103, y=xs103, fill_value='extrapolate')
		f107 = interp1d(x=erg107, y=xs107, fill_value='extrapolate')

		ixs103 = f103(erg16)
		ixs103mod = []
		for l in ixs103:
			if l < 0:
				ixs103mod.append(0)
			else:
				ixs103mod.append(l)

		ixs107 = f107(erg16)
		ixs107mod = []
		for l in ixs107:
			if l < 0:
				ixs107mod.append(0)
			else:
				ixs107mod.append(l)

		dummy_df = patient_library[patient_library.Z == nuclide[0]]
		dummy_df = dummy_df[dummy_df.A == nuclide[1]]
		dummy_df['MT103XS'] = ixs103mod
		dummy_df['MT107XS'] = ixs107mod

		new_df = pd.concat([new_df, dummy_df])

	# for xs103, i_erg  in zip(ixs103, erg16):
	# 	xs103_row = pd.Series({'Z': nuclide[0], 'A': nuclide[1], 'MT103XS': xs103, 'ERG': i_erg})
	# 	MT103_df = MT103_df._append(xs103_row, ignore_index=True)
	elif nuclide in nucs107 and nuclide not in nucs103:
		erg16, xs16 = General_plotter(df=patient_library, nuclides=[nuclide])
		erg107, xs107 = General_plotter(df=ENDFBVIII_MT107, nuclides=[nuclide])
		f107 = interp1d(x=erg107, y=xs107, fill_value='extrapolate')

		ixs107 = f107(erg16)
		ixs107mod = []
		for l in ixs107:
			if l < 0:
				ixs107mod.append(0)
			else:
				ixs107mod.append(l)

		dummy_df = patient_library[patient_library.Z == nuclide[0]]
		dummy_df = dummy_df[dummy_df.A == nuclide[1]]
		dummy_df['MT107XS'] = ixs107mod

		new_df = pd.concat([new_df, dummy_df])

	elif nuclide in nucs103 and nuclide not in nucs107:

		erg103, xs103 = General_plotter(df=ENDFBVIII_MT103, nuclides=[nuclide])
		erg16, xs16 = General_plotter(df=patient_library, nuclides=[nuclide])

		# nucmass[0].append(nuclide[1])
		# nucmass[1].append(max(xs103))
		# nucmass[2].append(np.mean(xs103))

		f103 = interp1d(x=erg103, y=xs103, fill_value='extrapolate')


		ixs103 = f103(erg16)
		ixs103mod = []
		for l in ixs103:
			if l < 0:
				ixs103mod.append(0)
			else:
				ixs103mod.append(l)

		dummy_df = patient_library[patient_library.Z == nuclide[0]]
		dummy_df = dummy_df[dummy_df.A == nuclide[1]]
		dummy_df['MT103XS'] = ixs103mod

		new_df = pd.concat([new_df, dummy_df])


	elif nuclide not in nucs103 and nuclide not in nucs107:
		dummy_df2 = patient_library[patient_library.Z == nuclide[0]]
		dummy_df2 = dummy_df2[dummy_df2.A == nuclide[1]]

		new_df = pd.concat([new_df, dummy_df2])






























# vis = new_df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'N', 'ME',
#        'BEA', 'BEA_A', 'AM', 'Sn', 'S2n', 'Sp', 'S2p', 'Radius', 'Pairing',
#        'Shell', 'Spin', 'Parity', 'Deform', 'Decay_Const', 'Z_even', 'A_even',
#        'N_even', 'beta_deformation', 'gamma_deformation',
#        'octopole_deformation', 'rms_radius', 'n_rms_radius', 'p_rms_radius',
#        'n_gap_erg', 'p_gap_erg', 'n_chem_erg', 'p_chem_erg', 'Sn_compound',
#        'S2n_compound', 'Sp_compound', 'S2p_compound', 'Radius_compound',
#        'ME_compound', 'BEA_compound', 'BEA_A_compound', 'Sn_daughter',
#        'S2n_daughter', 'Sp_daughter', 'S2p_daughter', 'Radius_daughter',
#        'ME_daughter', 'BEA_daughter', 'BEA_A_daughter', 'Pairing_compound',
#        'Shell_compound', 'Spin_compound', 'Parity_compound', 'Deform_compound',
#        'Decay_compound', 'Pairing_daughter', 'Shell_daughter', 'Spin_daughter',
#        'Parity_daughter', 'Deform_daughter', 'Decay_daughter',
#        'Nlow', 'Ulow', 'Ntop', 'Utop', 'ainf', 'XSupp', 'XSlow', 'Asymmetry',
#        'Asymmetry_compound', 'Asymmetry_daughter'])
# for u1, MT16_row in ENDFBVIII_MT16.iterrows():
# 	print(f"{u1}/{len(ENDFBVIII_MT16)}")
# 	if u1 > 15:
# 		break
#
# 	for u2, MT103_row in MT103_df.iterrows():
# 		if [MT103_row['Z'], MT103_row['A']] == [MT16_row['Z'], MT16_row['A']] and MT103_row['ERG'] == MT16_row['ERG']:
#
#
# 			dummy_row = MT16_row
# 			dummy_row['MT103XS'] = MT103_row['MT103XS']
# 			# MT16_row['MT103XS'] = MT103_row['MT103XS']
#
# 			new_df = new_df._append(dummy_row, ignore_index=True)
#
# 		# plt.figure()
# 		# plt.grid()
# 		# plt.plot(erg103, xs103, label = 'original')
# 		# plt.plot(erg16, ixs103, label = 'interpolated')
# 		# plt.legend()
# 		# plt.xlabel('Energy / MeV')
# 		# plt.ylabel('$\sigma_{n,p}$ / b')
# 		# plt.title(f"(n,p) cross sections for {periodictable.elements[nuclide[0]]}-{nuclide[1]}")
# 		# plt.show()
# 		# time.sleep(1)
# 		# count +=1
# 		# if count > 20:
# 		# 	break

# logxs = [np.log(i) for i in nucmass[2]]
# plt.figure()
# # plt.plot(nucmass[0], nucmass[1], label = 'max xs')
# plt.plot(nucmass[0], logxs, 'x', label = 'mean xs')
# plt.xlabel('A')
# plt.legend()
# plt.grid()
# plt.ylabel('XS / b')
# plt.show()
