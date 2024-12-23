import datetime
import warnings
import scipy.stats
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from MT_103_functions import *
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

df = pd.read_csv("ENDFBVIII_MT_103_all_features.csv")

energy_row = 4
min_energy = 0.01



df.index = range(len(df))


al = range_setter(la=0, ua=215, df=df)

TENDL = pd.read_csv("TENDL-2021_MT_103_all_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=210)


JEFF33 = pd.read_csv('JEFF-3.3_MT_103_all_features.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=0, ua=210)


JENDL5 = pd.read_csv('JENDL-5_MT_103_all_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL_nuclides = range_setter(df=JENDL5, la=0, ua=210)


CENDL32 = pd.read_csv('CENDL-3.2_MT_103_all_features.csv')
CENDL32.index = range(len(CENDL32))
CENDL_nuclides = range_setter(df=CENDL32, la=0, ua=210)


n_evaluations = 10
datapoint_matrix = []
target_nuclide = [42,95]

jendlerg, jendlxs = Generalplotter103(dataframe=JENDL5, nuclide=target_nuclide)
cendlerg, cendlxs = Generalplotter103(dataframe=CENDL32, nuclide=target_nuclide)
jefferg, jeffxs = Generalplotter103(dataframe=JEFF33, nuclide=target_nuclide)
tendlerg, tendlxs = Generalplotter103(dataframe=TENDL, nuclide=target_nuclide)
endfberg, endfbxs = Generalplotter103(dataframe=df, nuclide=target_nuclide)

runs_r2_array = []
runs_rmse_array = []

X_test, y_test = make_test([target_nuclide], df=df, minerg=min_energy, maxerg=20)

for i in tqdm.tqdm(range(n_evaluations)):
	# print(f"\nRun {i+1}/{n_evaluations}")

	validation_nuclides = [target_nuclide]
	validation_set_size = 10  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)

	time1 = time.time()

	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides,
								  la=30, ua=210, minerg=min_energy, maxerg=20, mode='dual')  # make training matrix


	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=1100,
								 learning_rate=0.008,
								 max_depth=8,
								 subsample=0.888,
								 reg_lambda=4,
								seed=model_seed
								 )

	model.fit(X_train, y_train)



	predictions = model.predict(X_test)

	if i == 0:
		for k, pred in enumerate(predictions):
			if [X_test[k,0], X_test[k,1]] == validation_nuclides[0]:
				datapoint_matrix.append([pred])
	else:
		valid_predictions = []
		for k, pred in enumerate(predictions):
			if [X_test[k, 0], X_test[k, 1]] == validation_nuclides[0]:
				valid_predictions.append(pred)
		for m, prediction in zip(datapoint_matrix, valid_predictions):
			m.append(prediction)

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
				dummy_test_E.append(row[energy_row]) # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	all_preds = []
	all_libs = []


	pred_endfb_r2 = r2_score(y_true=XS_plotmatrix[0], y_pred=P_plotmatrix[0])
	for x, y in zip(XS_plotmatrix[0], P_plotmatrix[0]):
		all_libs.append(y)
		all_preds.append(x)
	# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

	if target_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL32, minerg=min_energy, maxerg=20)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=pred_cendl)
		for x, y in zip(pred_cendl, cendl_xs):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - CENDL-3.2 R2: {pred_cendl_r2:0.5f}")

	if target_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL5, minerg=min_energy, maxerg=20)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=pred_jendl)
		# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")
		for x, y in zip(pred_jendl, jendl_xs):
			all_libs.append(y)
			all_preds.append(x)

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF33, minerg=min_energy, maxerg=20)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - JEFF-3.3 R2: {pred_jeff_r2:0.5f}")

	if target_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL, minerg=min_energy, maxerg=20)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=pred_tendl)
		for x, y in zip(pred_tendl, tendl_xs):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - TENDL-2021 R2: {pred_tendl_r2:0.5f}")
	consensus = r2_score(all_libs, all_preds)
	consensus_rmse = mean_squared_error(all_libs, all_preds) ** 0.5
	runs_r2_array.append(consensus)
	runs_rmse_array.append(consensus_rmse)

	# print(f"Consensus R2: {r2_score(all_libs, all_preds):0.5f}")

	exp_true_xs = [y for y in y_test]
	exp_pred_xs = [xs for xs in predictions]

	# print(f'completed in {time.time() - time1:0.1f} s')


XS_plot = []
E_plot = []

for i, row in enumerate(X_test):
	if [row[0], row[1]] == target_nuclide:
		XS_plot.append(y_test[i])
		E_plot.append(row[energy_row]) # Energy values are in 5th row


datapoint_means = []
datapoint_upper_interval = []
datapoint_lower_interval = []

d_l_1sigma = []
d_u_1sigma = []

for point in datapoint_matrix:

	mu = np.mean(point)
	sigma = np.std(point)
	interval_low, interval_high = scipy.stats.t.interval(0.95, loc=mu, scale=sigma, df=(len(point) - 1))
	# il_1sigma, ih_1sigma = scipy.stats.t.interval(0.68, loc=mu, scale=sigma, df=(len(point) - 1))
	datapoint_means.append(mu)
	datapoint_upper_interval.append(interval_high)
	datapoint_lower_interval.append(interval_low)

	# d_l_1sigma.append(il_1sigma)
	# d_u_1sigma.append(ih_1sigma)

lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point-low)
	upper_bound.append(point+up)







artemeve = [14.8]
artemevxs = [0.04]
artemevdxs = [0.005]

marcinkowskie = [13.4, 13.9, 14.5, 15.0, 15.4]
marcinkowskixs = [0.0428, 0.0375, 0.041, 0.0388, 0.0348]
marcinkowskidxs = [0.0051, 0.005, 0.0055, 0.0055, 0.0055]


rahmane = [5.911, 6.474, 7.032, 7.549, 8.132, 8.632, 9.153, 9.577]
rahmanxs = [0.00121, 0.00181, 0.00344, 0.0043, 0.00654, 0.01098, 0.014, 0.01634]
rahmandxs = [0.00015, 0.00023, 0.00043, 0.00055, 0.00084, 0.00141, 0.0017, 0.00195]

reimere = [13.48, 13.64, 13.87, 14.05, 14.28, 14.45, 14.64, 14.82]
reimerxs = [0.0345, 0.036, 0.0368, 0.039, 0.0399, 0.042, 0.0414, 0.0426]
reimerdxs = [0.0014, 0.0014, 0.0015, 0.0016, 0.0016, 0.0017, 0.0017, 0.0017]


semkovae = [8.08, 10.05, 12.09, 13.08]
semkovaxs = [0.0057, 0.0151, 0.0277, 0.036]
semkovadxs = [0.0008, 0.001, 0.0015, 0.001]

qaime = [14.7]
qaimxs = [0.031]
qaimdxs = [0.0036]

mollae = [13.57, 13.74, 14.1, 14.57, 14.71]
mollaxs = [0.02992, 0.02984, 0.0308, 0.03022, 0.03281]
molladxs = [0.0025, 0.00267, 0.00277, 0.00266, 0.00291]



#2sigma CF
plt.figure()
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendlerg, tendlxs, label = 'TENDL-2021', color='dimgrey')
if target_nuclide in JENDL_nuclides:
	plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')

if target_nuclide in CENDL_nuclides:
	plt.plot(cendlerg, cendlxs, '--', label = 'CENDL-3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
plt.errorbar(marcinkowskie, marcinkowskixs, yerr=marcinkowskidxs, fmt='x', color='indigo', capsize=2, label='Marcinkowski, 1986')
plt.errorbar(artemeve,artemevxs, yerr=artemevdxs, fmt='x', color='orangered', capsize=2, label='Artemev, 1980')
plt.errorbar(semkovae, semkovaxs, yerr=semkovadxs, fmt='x', color='mediumorchid', capsize=2, label='Semkova, 2014')
# plt.errorbar(ce, cxs, yerr=cdxs, fmt='x', color='gold', capsize=2, label='Chuanxin, 2011')
plt.errorbar(rahmane,rahmanxs, yerr=rahmandxs, fmt='x', color='dodgerblue', capsize=2, label='Rahman, 1985')
plt.errorbar(reimere, reimerxs, yerr=reimerdxs, fmt='x', color='firebrick', capsize=2, label='Reimer, 2005')
plt.errorbar(mollae,mollaxs, yerr=molladxs, fmt='x', color='slategrey', capsize=2, label='Molla, 1997')
# plt.errorbar(be, bxs, yerr=bdxs, fmt='x', color='purple', capsize=2, label = 'Bayhurst, 1975')
# plt.errorbar(frehaute, frehautxs, yerr=frehautdxs, fmt='x', color='violet', capsize=2, label = 'Frehaut, 1980')
# plt.errorbar(filae, filasexs, yerr=filasexsd, fmt='x', color='indigo', capsize=2, label = 'Filatenkov, 2016')
# plt.errorbar(cse, csxs, yerr=csxsd, fmt='x', color='violet',capsize=2, label='Csikai, 1967')
# plt.errorbar(tiwarie, tiwarixs, yerr=tiwarixsd,
# 			 fmt='x', color='indigo',capsize=2, label='Tiwari, 1968')
# plt.errorbar(hillmane, hillmanxs, yerr=hillmanxsd, fmt='x', color='orangered',capsize=2, label='Hillman, 1962')
#
# plt.errorbar(ikedae,ikedaxs, yerr=ikedaxsd, fmt='x',
# 			 capsize=2, label='Ikeda, 1988', color='blue')
# plt.errorbar(pue, puxs, yerr=puxsd, fmt='x', color='indigo',capsize=2, label='Pu, 2006')
# plt.errorbar(meghae, meghaxs, yerr=meghaxsd, fmt='x', color='violet',capsize=2, label='Megha, 2017')
# plt.errorbar(junhua_E, junhua_XS, yerr=junhua_XSd, fmt='x', color='orangered',capsize=2, label='Junhua, 2018')
# plt.errorbar(FrehautW182_E, FrehautW182_XS, yerr=FrehautW182_dXS,
# 			 fmt='x', color='indigo',capsize=2, label='Frehaut, 1980')
# plt.errorbar(Frehaut_Pb_E, Frehaut_Pb_XS, yerr=Frehaut_Pb_dXS, fmt='x', color='indigo',capsize=2, label='Frehaut, 1980')
# plt.errorbar(LuE, LuXS, yerr=LudXS, fmt='x', color='indigo',capsize=2, label='Lu, 1970')
# plt.errorbar(qaime,qaimxs, yerr=qaimdxs, fmt='x', color='lightgreen',capsize=2, label='Qaim, 1974')
# plt.errorbar(JunhuaE, JunhuaXS, yerr=JunhuadXS, fmt='x', color='blue',capsize=2, label='JunhuaLuoo, 2007')
# plt.errorbar(TemperleyE, TemperleyXS, yerr=TemperleydXS, fmt='x', color='orangered',capsize=2, label='Temperley, 1970')
# plt.errorbar(BormannE, BormannXS, yerr=BormanndXS, fmt='x',capsize=2, label='Bormann, 1970')

# plt.errorbar(Bayhurst_energies, Bayhurst_XS, Bayhurst_delta_XS, Bayhurst_delta_energies, fmt='x',
# 			 capsize=2, label='Bayhurst, 1975', color='indigo')
# plt.errorbar(Frehaut_E, Frehaut_XS, Frehaut_XS_d, Frehaut_E_d, fmt='x',
# 			 capsize=2, label='Frehaut, 1980', color='violet')
# plt.errorbar(Dzysiuk_energies, Dzysiuk_XS, Dzysiuk_delta_XS, Dzysiuk_delta_energies, fmt='x',
# 			 capsize=2, label='Dzysiuk, 2010', color='blue')
# plt.errorbar(Veeser_energies, Veeser_XS, Veeser_delta_XS, Veeser_delta_energies, fmt='x',
# 			 capsize=2, label='Veeser, 1977', color='orangered')

plt.grid()
plt.title(f"$\sigma_{{n,p}}$ for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,p}$ / b")
plt.legend(loc='upper left')
plt.show()

mu_r2 = np.mean(runs_r2_array)
sigma_r2 = np.std(runs_r2_array)
interval_low_r2, interval_high_r2 = scipy.stats.t.interval(0.95, loc=mu_r2, scale=sigma_r2, df=(len(runs_r2_array) - 1))
print(f"\nConsensus: {mu_r2:0.5f} +/- {interval_high_r2-interval_low_r2:0.5f}")

print(f'Consensus RMSE: {np.mean(runs_rmse_array):0.3f} +/- {np.std(runs_rmse_array):0.3f}')

final_runtime = time.time() - runtime



print()
print(f"Runtime: {timedelta(seconds=final_runtime)}")
