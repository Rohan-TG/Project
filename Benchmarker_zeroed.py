import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
import shap
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import anomaly_remover, make_train, make_test


# df = pd.read_csv('zeroed_1_xs_fund_feateng.csv') # new interpolated dataset, used for training only
# df_test = pd.read_csv('zeroed_1_xs_fund_feateng.csv') # original dataset, used for validation

df = pd.read_csv("level_densities_v1_zeroed_1_xs_fund_feateng.csv")
df_test = pd.read_csv("level_densities_v1_zeroed_1_xs_fund_feateng.csv")  # dataframe as above, but with the new features from the Gilbert-Cameron model

df_test = df_test[df_test.MT == 16] # extract (n,2n) only
df_test.index = range(len(df_test)) # re-label indices




df_test = anomaly_remover(dfa = df_test)

al = []
for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
	if [j, i] in al:
		continue
	else:
		al.append([j, i]) # format is [Z, A]


# cat_magic = []
#
# magic_numbers = [2, 8, 20, 28, 50, 82, 126]
#
#
# cat_magic_double_original = []
# cat_magic_neutron_original = []
# cat_magic_proton_original = []
#
# for z, n in zip(df_test['Z'], df_test['A']):
# 	if z in magic_numbers and n in magic_numbers:
# 		cat_magic_proton_original.append(1)
# 		cat_magic_double_original.append(1)
# 		cat_magic_neutron_original.append(1)
# 	elif z in magic_numbers and n not in magic_numbers:
# 		cat_magic_proton_original.append(1)
# 		cat_magic_neutron_original.append(0)
# 		cat_magic_double_original.append(0)
# 	elif z not in magic_numbers and n in magic_numbers:
# 		cat_magic_neutron_original.append(1)
# 		cat_magic_double_original.append(0)
# 		cat_magic_proton_original.append(0)
# 	else:
# 		cat_magic_proton_original.append(0)
# 		cat_magic_double_original.append(0)
# 		cat_magic_neutron_original.append(0)
#
# df_test.insert(78, 'cat_magic_proton', cat_magic_proton_original)
# df_test.insert(79, 'cat_magic_neutron', cat_magic_neutron_original)
# df_test.insert(80, 'cat_magic_double', cat_magic_double_original)
#
# df_test['cat_magic_proton'].astype('category')
# df_test['cat_magic_neutron'].astype('category')
# df_test['cat_magic_double'].astype("category")
#
#
# cat_magic_proton = []
# cat_magic_neutron = []
# cat_magic_double = []
#
# for z, n in zip(df['Z'], df['N']):
# 	if z in magic_numbers and n in magic_numbers:
# 		cat_magic_double.append(1)
# 		cat_magic_neutron.append(1)
# 		cat_magic_proton.append(1)
# 	elif z in magic_numbers and n not in magic_numbers:
# 		cat_magic_proton.append(1)
# 		cat_magic_neutron.append(0)
# 		cat_magic_double.append(0)
# 	elif z not in magic_numbers and n in magic_numbers:
# 		cat_magic_neutron.append(1)
# 		cat_magic_proton.append(0)
# 		cat_magic_double.append(0)
# 	else:
# 		cat_magic_proton.append(0)
# 		cat_magic_double.append(0)
# 		cat_magic_neutron.append(0)
#
#
# df.insert(78, 'cat_magic_proton', cat_magic_proton)
# df.insert(79, 'cat_magic_neutron', cat_magic_neutron)
# df.insert(80, 'cat_magic_double', cat_magic_double)
#
# df['cat_magic_proton'].astype('category')
# df['cat_magic_neutron'].astype('category')
# df['cat_magic_double'].astype("category")







benchmark_number = 21

nuclides_used = []

nuclide_mse = []
nuclide_r2 = []

if __name__ == "__main__":
	# lower_a = int(input("Enter lower boundary A: "))
	# upper_a = int(input("Enter upper boundary A: "))

	# overall_r2_list = []

	every_prediction_list = []
	every_true_value_list = []

	for i in range(benchmark_number):

		print(f"Epoch {i+1}/{benchmark_number}")

		validation_nuclides = []  # list of nuclides used for validation
		# test_nuclides = []
		validation_set_size = 20  # number of nuclides hidden from training

		while len(validation_nuclides) < validation_set_size:
			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
		print("Test nuclide selection complete")


		X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=260) # make training matrix

		X_test, y_test = make_test(validation_nuclides, df=df_test)

		# X_train must be in the shape (n_samples, n_features)
		# and y_train must be in the shape (n_samples) of the target

		print("Train/val matrices generated")


		model = xg.XGBRegressor(n_estimators=900,
								learning_rate=0.01555,
								max_depth=8,
								subsample=0.18236,
								max_leaves=0,
								seed=42,)

		time1 = time.time()
		model.fit(X_train, y_train)

		print("Training complete")

		predictions = model.predict(X_test) # XS predictions


		# Form arrays for plots below
		XS_plotmatrix = []
		E_plotmatrix = []
		P_plotmatrix = []
		for nuclide in validation_nuclides:
			dummy_test_XS = []
			dummy_test_E = []
			dummy_predictions = []
			for i, row in enumerate(X_test):
				if [row[0], row[1]] == nuclide:
					dummy_test_XS.append(y_test[i])
					dummy_test_E.append(row[4]) # Energy values are in 5th row
					dummy_predictions.append(predictions[i])

			XS_plotmatrix.append(dummy_test_XS)
			E_plotmatrix.append(dummy_test_E)
			P_plotmatrix.append(dummy_predictions)

		# plot predictions against data
		for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
			nuc = validation_nuclides[i] # validation nuclide
			# plt.plot(erg, pred_xs, label='predictions')
			# plt.plot(erg, true_xs, label='data')
			# plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
			# plt.legend()
			# plt.grid()
			# plt.ylabel('XS / b')
			# plt.xlabel('Energy / MeV')
			# plt.show()
			#
			r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
			print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")

			mse = mean_squared_error(true_xs, pred_xs)

			nuclide_mse.append([nuc[0], nuc[1], mse])
			nuclide_r2.append([nuc[0], nuc[1], r2])
			# individual_r2_list.append(r2)

		overall_r2 = r2_score(y_test, predictions)
		for pred in predictions:
			every_prediction_list.append(pred)

		for val in y_test:
			every_true_value_list.append(val)
		# overall_r2_list.append(overall_r2)
		# print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
		print(f"R2: {overall_r2}") # Total R^2 for all predictions in this training campaign
		print(f'completed in {time.time() - time1} s \n')

		# explainer = shap.Explainer(model.predict, X_train,
		# 						   feature_names= ['Z', 'A', 'S2n', 'S2p', 'E', 'Sp', 'Sn',
		# 										   'BEA', 'P', 'Snc', 'g-def', 'N',
		# 										   'b-def',
		# 										   'Sn_da',
		# 										   'Sp_d',
		# 										   'S2n_d',
		# 										   'Radius',
		# 										   'n_g_erg',
		# 										   'n_c_erg',
		# 										   # 'xsmax',
		# 										   'n_rms_r',
		# 										   # 'oct_def',
		# 										   'D_c',
		# 										   'BEA_d',
		# 										   'BEA_c',
		# 										   'Pair_d',
		# 										   # 'Par_d',
		# 										   'S2n_c',
		# 										   'S2p_c',
		# 										   'ME',
		# 										   # 'Z_even',
		# 										   # 'A_even',
		# 										   # 'N_even',
		# 										   'Shell',
		# 										   # 'Par',
		# 										   'Spin',
		# 										   'Decay',
		# 										   'Deform',
		# 										   'p_g_e',
		# 										   'p_c_e',
		# 										   'p_rms_r',
		# 										   'rms_r',
		# 										   'Sp_c',
		# 										   'S_n_c',
		# 										   'Shell_c',
		# 										   # 'S2p-d',
		# 										   'Shell-d',
		# 										   'Spin-c',
		# 										   'Rad-c',
		# 										   'Def-c',
		# 										   'ME-c',
		# 										   'BEA-A-c',
		# 										   'Decay-d',
		# 										   'ME-d',
		# 										   'Rad-d',
		# 										   'Pair-c',
		# 										   # 'Par-c',
		# 										   'BEA-A-d',
		# 										   # 'Spin-d',
		# 										   'Def-d',
		# 										   'mag_p',
		# 										   'mag_n',
		# 										   'mag_d',
		# 										   ]) # SHAP feature importance analysis
		# shap_values = explainer(X_test)

		# name features for FIA
		# model.get_booster().feature_names = ['Z', 'A', 'S2n', 'S2p', 'E', 'Sp', 'Sn',
		# 									 'BEA',
		# 									 # 'P',
		# 									 'Snc', 'g-def',
		# 									 'N',
		# 									 'b-def',
		# 									 'Sn da',
		# 									 'Sp d',
		# 									 'S2n d',
		# 									 'Radius',
		# 									 'n_g_erg',
		# 									 'n_c_erg',
		# 									 # 'xsmax',
		# 									 'n_rms_r',
		# 									 # 'oct_def',
		# 									 'D_c',
		# 									 'BEA_d',
		# 									 'BEA_c',
		# 									 # 'Pair_d',
		# 									 # 'Par_d',
		# 									 'S2n_c',
		# 									 'S2p_c',
		# 									 'ME',
		# 									 # 'Z_even',
		# 									 # 'A_even',
		# 									 # 'N_even',
		# 									 'Shell',
		# 									 'Parity',
		# 									 'Spin',
		# 									 'Decay',
		# 									 'Deform',
		# 									 'p_g_e',
		# 									 'p_c_e',
		# 									 'p_rms_r',
		# 									 'rms_r',
		# 									 'Sp_c',
		# 									 # 'Sn_c',
		# 									 'Shell_c',
		# 									 # 'S2p-d',
		# 									 # 'Shell-d',
		# 									 'Spin-c',
		# 									 'Rad-c',
		# 									 'Def-c',
		# 									 'ME-c',
		# 									 'BEA-A-c',
		# 									 'Decay-d',
		# 									 'ME-d',
		# 									 # 'Rad-d',
		# 									 # 'Pair-c',
		# 									 # 'Par-c',
		# 									 'BEA-A-d',
		# 									 # 'Spin-d',
		# 									 'Def-d',
		# 									 # 'mag_p',
		# 									 # 'mag-n',
		# 									 # 'mag-d',
		# 									 'Nlow',
		# 									 'Ulow',
		# 									 'Ntop',
		# 									 'Utop',
		# 									 'ainf',
		# 									 ]
		#
		# plt.figure(figsize=(10,12))
		# xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60) # metric is total gain
		# plt.show()

		# shap.plots.bar(shap_values, max_display = 70) # display SHAP results
		# shap.plots.waterfall(shap_values[0], max_display=70)



print()

print(f"New overall r2: {r2_score(every_true_value_list, every_prediction_list)}")



A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]
# mse_log_plots = [np.log(i[-1]) for i in nuclide_mse]
# mse_plots = [i[-1] for i in nuclide_mse]

log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]

# plt.figure()
# plt.plot(A_plots, mse_log_plots, 'x')
# plt.xlabel("A")
# plt.ylabel("log MSE")
# plt.title("log MSE")
# plt.show()
#
# plt.figure()
# plt.plot(A_plots, mse_plots, 'x')
# plt.xlabel("A")
# plt.ylabel("MSE")
# plt.title("Normal MSE")
# plt.show()

plt.figure()
plt.plot(A_plots, log_plots, 'x')
plt.xlabel("A")
plt.ylabel("log abs r2")
plt.title("log abs r2")
plt.grid()
plt.show()

plt.figure()
plt.plot(Z_plots, log_plots_Z, 'x')
plt.xlabel('Z')
plt.ylabel("log abs r2")
plt.title("log abs r2 - Z")
plt.grid()
plt.show()