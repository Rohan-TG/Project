import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
import tqdm
import scipy
from sklearn.metrics import r2_score
from matrix_functions import make_test, make_train, range_setter, General_plotter, feature_fetcher

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")


TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=4, ua=270)

JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=4, ua=270)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=4, ua=270)

CENDL = pd.read_csv('CENDL32_features_arange_zeroed.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=4, ua=270)


al = range_setter(la=4, ua=270, df=df)



nuclide_mse = []
nuclide_r2 = []


every_prediction_list = []
every_true_value_list = []

feature_matrix_Sn = []
feature_matrix_Asymmetry_daughter = []
# feature_matrix_BEA_A_compound = []
# feature_matrix_ME = []
feature_matrix_N = []
# feature_matrix_n_rms_radius = [] # 111 missing
feature_matrix_S2n = []
feature_matrix_Asymmetry = []
feature_matrix_BEA_A_daughter = []
# feature_matrix_n_chem_erg = [] # 111 missing
# feature_matrix_Radius = [] # 8253 missing
feature_matrix_S2p = []
# feature_matrix_Utop = [] # 16919 missing
feature_matrix_Shell = []
feature_matrix_Decay_Const = []
# feature_matrix_Spin = [] # 61 missing
# feature_matrix_Parity = [] # 61 missing


for nuclide in al:
	feat = feature_fetcher(feature='Sn', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_Sn.append([feat])

	feat = feature_fetcher(feature='Asymmetry_daughter', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_Asymmetry_daughter.append([feat])

	# feat = feature_fetcher(feature='BEA_A_compound', df=TENDL, z=nuclide[0], a=nuclide[1])
	# feature_matrix_BEA_A_compound.append([feat])

	# feat = feature_fetcher(feature='ME', df=TENDL, z=nuclide[0], a=nuclide[1])
	# feature_matrix_ME.append([feat])

	feat = feature_fetcher(feature='N', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_N.append([feat])

	feat = feature_fetcher(feature='S2n', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_S2n.append([feat])

	feat = feature_fetcher(feature='Asymmetry', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_Asymmetry.append([feat])

	feat = feature_fetcher(feature='BEA_A_daughter', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_BEA_A_daughter.append([feat])

	feat = feature_fetcher(feature='S2p', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_S2p.append([feat])

	feat = feature_fetcher(feature='Shell', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_Shell.append([feat])

	feat = feature_fetcher(feature='Decay_Const', df=df, z=nuclide[0], a=nuclide[1])
	feature_matrix_Decay_Const.append([feat])


n_runs = 20

for idx in tqdm.tqdm(range(n_runs)):
	nuclides_used = []
	counter = 0
	while len(nuclides_used) < len(al):

		validation_nuclides = []  # list of nuclides used for validation
		# test_nuclides = []
		validation_set_size = 20  # number of nuclides hidden from training

		while len(validation_nuclides) < validation_set_size:

			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
			if len(nuclides_used) == len(al):
				break


		print("Test nuclide selection complete")
		print(f"{len(nuclides_used)}/{len(al)} selections")


		X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides,
										  la=4, ua=270) # make training matrix

		X_test, y_test = make_test(validation_nuclides, df=df)

		print("Training...")


		model = xg.XGBRegressor(n_estimators=900,
								learning_rate=0.01,
								max_depth=8,
								subsample=0.18236,
								max_leaves=0,
								seed=42,)


		time1 = time.time()
		model.fit(X_train, y_train)

		predictions = model.predict(X_test) # XS predictions

		print("Processing evaluations...")

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

		batch_r2 = []
		for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
			nuc = validation_nuclides[i]

			evaluation_r2s = []
			evaluation_r2s.append(r2_score(y_true=true_xs, y_pred=pred_xs))
			# if nuc in al:
			# 	endfb8_erg, endfb8_xs = General_plotter(df=df, nuclides=[nuc])
			#
			# 	x_interpolate_endfb8 = np.linspace(start=0.00000000, stop=max(endfb8_erg), num=500)
			# 	f_pred_endfb8 = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
			# 	predictions_interpolated_endfb8 = f_pred_endfb8(x_interpolate_endfb8)
			#
			# 	f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg, y=endfb8_xs, fill_value='extrapolate')
			# 	endfb8_interp_xs = f_endfb8(x_interpolate_endfb8)
			# 	pred_endfb_r2 = r2_score(y_true=endfb8_interp_xs, y_pred=predictions_interpolated_endfb8)
			# 	evaluation_r2s.append(pred_endfb_r2)
			# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

			if nuc in CENDL_nuclides:
				cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[nuc])

				x_interpolate_cendl = np.linspace(start=0.000000001, stop=max(cendl_erg), num=500)
				f_pred_cendl = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
				predictions_interpolated_cendl = f_pred_cendl(x_interpolate_cendl)

				f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs)
				cendl_interp_xs = f_cendl32(x_interpolate_cendl)
				pred_cendl_r2 = r2_score(y_true=cendl_interp_xs, y_pred=predictions_interpolated_cendl)
				evaluation_r2s.append(pred_cendl_r2)
			# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

			if nuc in JENDL_nuclides:
				jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[nuc])

				x_interpolate_jendl = np.linspace(start=0.000000001, stop=max(jendl_erg), num=500)
				f_pred_jendl = scipy.interpolate.interp1d(x=erg, y=pred_xs)
				predictions_interpolated_jendl = f_pred_jendl(x_interpolate_jendl)

				f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs)
				jendl_interp_xs = f_jendl5(x_interpolate_jendl)
				pred_jendl_r2 = r2_score(y_true=jendl_interp_xs, y_pred=predictions_interpolated_jendl)
				evaluation_r2s.append(pred_jendl_r2)
			# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

			if nuc in JEFF_nuclides:
				jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[nuc])

				x_interpolate_jeff = np.linspace(start=0.000000001, stop=max(jeff_erg), num=500)
				f_pred_jeff = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
				predictions_interpolated_jeff = f_pred_jeff(x_interpolate_jeff)

				f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs)
				jeff_interp_xs = f_jeff33(x_interpolate_jeff)
				pred_jeff_r2 = r2_score(y_true=jeff_interp_xs, y_pred=predictions_interpolated_jeff)
				evaluation_r2s.append(pred_jeff_r2)

			if nuc in TENDL_nuclides:
				tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[nuc])

				x_interpolate_tendl = np.linspace(start=0.000000001, stop=max(tendl_erg), num=500)
				f_pred_tendl = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
				predictions_interpolated_tendl = f_pred_tendl(x_interpolate_tendl)

				f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs)
				tendl_interp_xs = f_tendl21(x_interpolate_tendl)
				pred_tendl_r2 = r2_score(y_true=tendl_interp_xs, y_pred=predictions_interpolated_tendl)
				evaluation_r2s.append(pred_tendl_r2)
			# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

			mean_nuclide_r2 = np.mean(evaluation_r2s) # r2 for nuclide nuc


			nuclide_r2.append([nuc[0], nuc[1], mean_nuclide_r2])

			feat_Sn = feature_fetcher(feature='Sn', df=df, z=nuc[0], a=nuc[1])

			feat_Asymmetry_daughter = feature_fetcher(feature='Asymmetry_daughter', df=df, z=nuc[0], a=nuc[1])

			# feat_BEA_A_compound = feature_fetcher(feature='BEA_A_compound', df=df, z=nuc[0], a=nuc[1])

			# feat_ME = feature_fetcher(feature='ME', df=df, z=nuc[0], a=nuc[1])

			feat_N = feature_fetcher(feature='N', df=df, z=nuc[0], a=nuc[1])

			feat_S2n = feature_fetcher(feature='S2n', df=df, z=nuc[0], a=nuc[1])

			feat_Asymmetry = feature_fetcher(feature='Asymmetry', df=df, z=nuc[0], a=nuc[1])

			feat_BEA_A_daughter = feature_fetcher(feature='BEA_A_daughter', df=df, z=nuc[0], a=nuc[1])

			feat_S2p = feature_fetcher(feature='S2p', df=df, z=nuc[0], a=nuc[1])

			feat_Shell = feature_fetcher(feature='Shell', df=df, z=nuc[0], a=nuc[1])

			feat_Decay_Const = feature_fetcher(feature='Decay_Const', df=df, z=nuc[0], a=nuc[1])

			for i, ft in enumerate(feature_matrix_Sn):
				if ft[0] == feat_Sn:
					feature_matrix_Sn[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_Asymmetry_daughter):
				if ft[0] == feat_Asymmetry_daughter:
					feature_matrix_Asymmetry_daughter[i].append(mean_nuclide_r2)

			# for i, ft in enumerate(feature_matrix_BEA_A_compound):
			# 	if ft[0] == feat_BEA_A_compound:
			# 		feature_matrix_BEA_A_compound[i].append(mean_nuclide_r2)

			# for i, ft in enumerate(feature_matrix_ME):
			# 	if ft[0] == feat_ME:
			# 		feature_matrix_ME[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_N):
				if ft[0] == feat_N:
					feature_matrix_N[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_S2n):
				if ft[0] == feat_S2n:
					feature_matrix_S2n[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_Asymmetry):
				if ft[0] == feat_Asymmetry:
					feature_matrix_Asymmetry[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_BEA_A_daughter):
				if ft[0] == feat_BEA_A_daughter:
					feature_matrix_BEA_A_daughter[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_S2p):
				if ft[0] == feat_S2p:
					feature_matrix_S2p[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_Shell):
				if ft[0] == feat_Shell:
					feature_matrix_Shell[i].append(mean_nuclide_r2)

			for i, ft in enumerate(feature_matrix_Decay_Const):
				if ft[0] == feat_Decay_Const:
					feature_matrix_Decay_Const[i].append(mean_nuclide_r2)


		for val in y_test:
			every_true_value_list.append(val)
		time_taken = time.time() - time1
		print(f'completed in {time_taken:0.1f} s.\n')

	print(f"Completed run {idx+1}/{n_runs}")


print()
print(feature_matrix_Asymmetry)

# print(f"New overall r2: {r2_score(every_true_value_list, every_prediction_list)}")

A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]


log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]

r2_mean_Sn = []
features_only_Sn = []
for i in feature_matrix_Sn:
	dum = i[1:]
	feature = i[0]
	features_only_Sn.append(feature)
	avg = np.mean(dum)
	r2_mean_Sn.append(avg)

log_r2_mean_Sn = [abs(np.log(abs(i))) for i in r2_mean_Sn]



r2_mean_Asymmetry_daughter = []
features_only_Asymmetry_daughter = []
for i in feature_matrix_Asymmetry_daughter:
	dum = i[1:]
	feature = i[0]
	features_only_Asymmetry_daughter.append(feature)
	avg = np.mean(dum)
	r2_mean_Asymmetry_daughter.append(avg)

log_r2_mean_Asymmetry_daughter = [abs(np.log(abs(i))) for i in r2_mean_Asymmetry_daughter]



# r2_mean_BEA_A_compound = []
# features_only_BEA_A_compound = []
# for i in feature_matrix_BEA_A_compound:
# 	dum = i[1:]
# 	feature = i[0]
# 	features_only_BEA_A_compound.append(feature)
# 	avg = np.mean(dum)
# 	r2_mean_BEA_A_compound.append(avg)
#
# log_r2_mean_BEA_A_compound = [abs(np.log(abs(i))) for i in r2_mean_Sn]



# r2_mean_ME = []
# features_only_ME = []
# for i in feature_matrix_ME:
# 	dum = i[1:]
# 	feature = i[0]
# 	features_only_ME.append(feature)
# 	avg = np.mean(dum)
# 	r2_mean_ME.append(avg)
#
# log_r2_mean_ME = [abs(np.log(abs(i))) for i in r2_mean_ME]


r2_mean_N = []
features_only_N = []
for i in feature_matrix_N:
	dum = i[1:]
	feature = i[0]
	features_only_N.append(feature)
	avg = np.mean(dum)
	r2_mean_N.append(avg)

log_r2_mean_N = [abs(np.log(abs(i))) for i in r2_mean_N]



r2_mean_S2n = []
features_only_S2n = []
for i in feature_matrix_S2n:
	dum = i[1:]
	feature = i[0]
	features_only_S2n.append(feature)
	avg = np.mean(dum)
	r2_mean_S2n.append(avg)

log_r2_mean_S2n = [abs(np.log(abs(i))) for i in r2_mean_S2n]



r2_mean_Asymmetry = []
features_only_Asymmetry = []
for i in feature_matrix_Asymmetry:
	dum = i[1:]
	feature = i[0]
	features_only_Asymmetry.append(feature)
	avg = np.mean(dum)
	r2_mean_Asymmetry.append(avg)

log_r2_mean_Asymmetry = [abs(np.log(abs(i))) for i in r2_mean_Asymmetry]



r2_mean_BEA_A_daughter = []
features_only_BEA_A_daughter = []
for i in feature_matrix_BEA_A_daughter:
	dum = i[1:]
	feature = i[0]
	features_only_BEA_A_daughter.append(feature)
	avg = np.mean(dum)
	r2_mean_BEA_A_daughter.append(avg)

log_r2_mean_BEA_A_daughter = [abs(np.log(abs(i))) for i in r2_mean_BEA_A_daughter]


r2_mean_S2p = []
features_only_S2p = []
for i in feature_matrix_S2p:
	dum = i[1:]
	feature = i[0]
	features_only_S2p.append(feature)
	avg = np.mean(dum)
	r2_mean_S2p.append(avg)

log_r2_mean_S2p = [abs(np.log(abs(i))) for i in r2_mean_S2p]


r2_mean_Shell = []
features_only_Shell = []
for i in feature_matrix_Shell:
	dum = i[1:]
	feature = i[0]
	features_only_Shell.append(feature)
	avg = np.mean(dum)
	r2_mean_Shell.append(avg)

log_r2_mean_Shell = [abs(np.log(abs(i))) for i in r2_mean_Shell]


r2_mean_Decay_Const = []
features_only_Decay_Const = []
for i in feature_matrix_Decay_Const:
	dum = i[1:]
	feature = i[0]
	features_only_Decay_Const.append(feature)
	avg = np.mean(dum)
	r2_mean_Decay_Const.append(avg)

log_r2_mean_Decay_Const = [abs(np.log(abs(i))) for i in r2_mean_Decay_Const]

# Sn
plt.figure()
plt.plot(features_only_Sn, log_r2_mean_Sn, 'x')
plt.xlabel('Sn')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - Sn')
plt.grid()
plt.show()

# Asymmetry_daughter
plt.figure()
plt.plot(features_only_Asymmetry_daughter, log_r2_mean_Asymmetry_daughter, 'x')
plt.xlabel('Daughter Asymmetry')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - Daughter Asymmetry')
plt.grid()
plt.show()

# plt.figure()
# plt.plot(features_only_BEA_A_compound, log_r2_mean_BEA_A_compound, 'x')
# plt.xlabel('Sn')
# plt.ylabel('$|\ln(|r^2|)|$')
# plt.title('$|\ln(|r^2|)|$ Performance - BEA_A Compound')
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(features_only_ME, log_r2_mean_ME, 'x')
# plt.xlabel('Sn')
# plt.ylabel('$|\ln(|r^2|)|$')
# plt.title('$|\ln(|r^2|)|$ Performance - Mass Excess')
# plt.grid()
# plt.show()
# time.sleep(2)

plt.figure()
plt.plot(features_only_N, log_r2_mean_N, 'x')
plt.xlabel('N')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - N')
plt.grid()
plt.show()

plt.figure()
plt.plot(features_only_S2n, log_r2_mean_S2n, 'x')
plt.xlabel('S2n')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - S2n')
plt.grid()
plt.show()

plt.figure()
plt.plot(features_only_Asymmetry, log_r2_mean_Asymmetry, 'x')
plt.xlabel('Asymmetry')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - Asymmetry')
plt.grid()
plt.show()
time.sleep(1)

plt.figure()
plt.plot(features_only_BEA_A_daughter, log_r2_mean_BEA_A_daughter, 'x')
plt.xlabel('BEA_A Daughter')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - BEA_A Daughter')
plt.grid()
plt.show()

plt.figure()
plt.plot(features_only_S2p, log_r2_mean_S2p, 'x')
plt.xlabel('S2p')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - S2p')
plt.grid()
plt.show()


plt.figure()
plt.plot(features_only_Shell, log_r2_mean_Shell, 'x')
plt.xlabel('Shell')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - Shell')
plt.grid()
plt.show()
time.sleep(2)


log_decay_constant = [np.log(dec) for dec in features_only_Decay_Const]
plt.figure()
plt.plot(log_decay_constant, log_r2_mean_Decay_Const, 'x')
plt.xlabel('ln(Decay Constant)')
plt.ylabel('$|\ln(|r^2|)|$')
plt.title('$|\ln(|r^2|)|$ Performance - Decay Constant')
plt.grid()
plt.show()


plt.figure()
plt.plot(A_plots, log_plots, 'x')
plt.xlabel("A")
plt.ylabel("log abs r2")
plt.title("$|\ln(|r^2|)|$ Performance - A")
plt.grid()
plt.show()

plt.figure()
plt.plot(Z_plots, log_plots_Z, 'x')
plt.xlabel('Z')
plt.ylabel("log abs r2")
plt.title("$|\ln(|r^2|)|$ Performance - Z")
plt.grid()
plt.show()