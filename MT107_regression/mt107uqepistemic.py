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
from functions107 import  maketest107, maketrain107, Generalplotter107
from matrix_functions import range_setter, r2_standardiser
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

df = pd.read_csv("ENDFBVIII_MT_107_all_features.csv")


df.index = range(len(df))


al = range_setter(la=30, ua=215, df=df)


TENDL = pd.read_csv("TENDL-2021_MT_107_all_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)


JEFF33 = pd.read_csv('JEFF-3.3_MT_107_all_features.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=210)


JENDL5 = pd.read_csv('JENDL-5_MT_107_all_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=210)


CENDL32 = pd.read_csv('CENDL-3.2_MT_107_all_features.csv')
CENDL32.index = range(len(CENDL32))
CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=210)


n_evaluations = 100
datapoint_matrix = []
target_nuclide = [42,100]

jendlerg, jendlxs = Generalplotter107(JENDL5, target_nuclide)
cendlerg, cendlxs = Generalplotter107(CENDL32,target_nuclide)
jefferg, jeffxs = Generalplotter107(JEFF33, target_nuclide)
tendlerg, tendlxs = Generalplotter107(TENDL, target_nuclide)
endfberg, endfbxs = Generalplotter107(df, target_nuclide)

jendlr2s= []
cendlr2s = []
endfbr2s = []
jeffr2s = []
tendlr2s = []

jendlrmses = []
cendlrmses = []
tendlrmses = []
jeffrmses = []
endfbrmses = []

runs_r2_array = []
runs_rmse_array = []

X_test, y_test = maketest107([target_nuclide], df=df, )

print('Data loaded.')
for i in tqdm.tqdm(range(n_evaluations)):
	# print(f"\nRun {i+1}/{n_evaluations}")

	validation_nuclides = [target_nuclide]
	validation_set_size = 1  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	# print("\nTest nuclide selection complete")

	time1 = time.time()

	X_train, y_train = maketrain107(df=df, validation_nuclides=validation_nuclides,
								  la=30, ua=208, )  # make training matrix

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.01,
							max_depth=7,
							subsample=0.5,
							reg_lambda=2,
							seed=model_seed
							)

	model.fit(X_train, y_train)

	predictions_ReLU = []

	for n in validation_nuclides:

		temp_x, temp_y = maketest107(nuclides=[n], df=df)
		initial_predictions = model.predict(temp_x)

		for p in initial_predictions:
			if p >= (0.02 * max(initial_predictions)):
				predictions_ReLU.append(p)
			else:
				predictions_ReLU.append(0.0)


	predictions = predictions_ReLU

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
				dummy_test_E.append(row[4]) # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	all_preds = []
	all_libs = []


	d1, d2, pred_endfb_r2 = r2_standardiser(library_xs=XS_plotmatrix[0],predicted_xs=P_plotmatrix[0])
	endfbr2s.append(pred_endfb_r2)
	for x, y in zip(d1, d2):
		all_libs.append(y)
		all_preds.append(x)
	# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

	if target_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = maketest107(nuclides=[target_nuclide], df=CENDL32)
		pred_cendl = model.predict(cendl_test)

		d1, d2,pred_cendl_r2 = r2_standardiser(library_xs=cendl_xs, predicted_xs=pred_cendl)
		cendlr2s.append(pred_cendl_r2)
		for x, y in zip(d1, d2):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - CENDL-3.2 R2: {pred_cendl_r2:0.5f}")

	if target_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = maketest107(nuclides=[target_nuclide], df=JENDL5)
		pred_jendl = model.predict(jendl_test)

		gatedpredjendl, gatedjendl, pred_jendl_r2 = r2_standardiser(library_xs=jendl_xs, predicted_xs=pred_jendl)
		jendlr2s.append(pred_jendl_r2)
		# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")
		for x, y in zip(gatedpredjendl, gatedjendl):
			all_libs.append(y)
			all_preds.append(x)

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = maketest107(nuclides=[target_nuclide], df=JEFF33)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(pred_jeff, jeff_xs)
		jeffr2s.append(pred_jeff_r2)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - JEFF-3.3 R2: {pred_jeff_r2:0.5f}")

	if target_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = maketest107(nuclides=[target_nuclide], df=TENDL)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		dp,d2,pred_tendl_r2 = r2_standardiser(library_xs=tendl_xs, predicted_xs=pred_tendl)
		tendlr2s.append(pred_tendl_r2)
		for x, y in zip(dp, d2):
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
		E_plot.append(row[4]) # Energy values are in 5th row


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


ikedae = [13.35, 13.55, 13.55, 13.58, 13.68, 13.76, 13.81, 13.81, 13.95, 13.95, 13.99, 14.1, 14.23, 14.32, 14.44, 14.53, 14.53, 14.71, 14.71, 14.84, 14.92, 14.92, 14.95, 14.96, 14.96]
ikedaxs = [0.00211, 0.00254, 0.00202, 0.00215, 0.00209, 0.00227, 0.00218, 0.00259, 0.00237, 0.00233, 0.00259, 0.00202, 0.0029, 0.00223, 0.00281, 0.00196, 0.00287, 0.00284, 0.00305, 0.00286, 0.00268, 0.00306, 0.00339, 0.00307, 0.00277]
ikedadxs = [0.00017, 0.00041, 0.00023, 0.00017, 0.00021, 0.00017, 0.00031, 0.00023, 0.00034, 0.00023, 0.00019, 0.00022, 0.00021, 0.00027, 0.0002, 0.00023, 0.00023, 0.00028, 0.00023, 0.00027, 0.00025, 0.00023, 0.00023, 0.00026, 0.00021]

liskiene = [12.59, 12.64, 13.39, 13.43, 14.36, 14.39, 15.82, 15.84, 16.9, 16.96, 17.92, 17.99, 18.81, 18.88, 19.58]
liskienxs = [0.002, 0.0027, 0.0017, 0.002, 0.0035, 0.0045, 0.0033, 0.0044, 0.0052, 0.0052, 0.0053, 0.0043, 0.0048, 0.0047, 0.0047]
liskiendxs = [0.0005, 0.0005, 0.0005, 0.0005, 0.0006, 0.0006, 0.0005, 0.0007, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0004169]


marcinkowskie = [13.0, 13.4, 13.9, 14.5, 15.0, 15.4, 15.9, 16.6]
marcinkowskixs = [0.001, 0.0021, 0.0025, 0.0027, 0.0033, 0.0037, 0.0026, 0.0026]
marcinkowskidxs = [0.0001, 0.0002, 0.0003, 0.0003, 0.0003, 0.0005, 0.0002, 0.0003]


reimere = [16.2, 18, 19.3, 20.5]
reimerxs = [0.0034, 0.0046, 0.0053, 0.0052]
reimerdxs = [0.0002, 0.0004, 0.0005, 0.0005]


semkovae = [1.09200E+01, 1.20900E+01, 1.30800E+01, 1.44800E+01]
semkovaxs = [7.20000E-01 / 1000, 1.12000E+00 / 1000, 1.78000E+00 / 1000, 3.10000E+00 / 1000]


semkovadxs = [1.40000E-01 / 1000, 1.00000E-01 / 1000, 1.00000E-01 / 1000, 2.00000E-01 / 1000]

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
plt.errorbar(reimere, reimerxs, yerr=reimerdxs, fmt='x', color='indigo', label= 'Reimer, 2005', capsize=2)
plt.errorbar(semkovae, semkovaxs, yerr=semkovadxs, fmt='x', color = 'orange', label= 'Semkova, 2014', capsize=2)
plt.errorbar(marcinkowskie, marcinkowskixs, yerr=marcinkowskidxs, fmt='x', color = 'tomato', label= 'Marcinkowski, 1986', capsize=2)
plt.errorbar(ikedae, ikedaxs, yerr=ikedadxs, fmt='x', color = 'lawngreen', label= 'Ikeda, 1988', capsize=2)
plt.errorbar(liskiene, liskienxs, yerr=liskiendxs, fmt='x', color = 'blue', label= 'Liskien, 1990', capsize=2)
# plt.errorbar(fe, fxs, yerr=fdxs, fmt='x', color='indigo', capsize=2, label='Frehaut, 1980')
# plt.errorbar(ne, nxs, yerr=ndxs, fmt='x', color='orangered', capsize=2, label='Nethaway, 1972')
# plt.errorbar(fe, fxs, yerr=fdxs, fmt='x', color='magenta', capsize=2, label='Frehaut, 1980')
# plt.errorbar(ce, cxs, yerr=cdxs, fmt='x', color='gold', capsize=2, label='Chuanxin, 2011')
# plt.errorbar(de, dxs, yerr=ddxs, fmt='x', color='dodgerblue', capsize=2, label='Dzysiuk, 2010')
# plt.errorbar(ve, vxs, yerr=vdxs, fmt='x', color='firebrick', capsize=2, label='Veeser, 1977')
# plt.errorbar(ze, zxs, yerr=zdxs, fmt='x', color='aquamarine', capsize=2, label='Zhengwei, 2018')
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
# plt.errorbar(QaimE, QaimXS, yerr=QaimdXS, fmt='x', color='violet',capsize=2, label='Qaim, 1974')
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
plt.title(fr"$\sigma_{{n,\alpha}}$ for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.xlabel("Energy / MeV")
plt.ylabel(r"$\sigma_{n,\alpha}$ / b")
plt.legend(loc='upper left')
plt.show()

mu_r2 = np.mean(runs_r2_array)
sigma_r2 = np.std(runs_r2_array)
interval_low_r2, interval_high_r2 = scipy.stats.t.interval(0.95, loc=mu_r2, scale=sigma_r2, df=(len(runs_r2_array) - 1))
print(f"\nConsensus: {mu_r2:0.5f} +/- {interval_high_r2-interval_low_r2:0.5f}")

print(f'Consensus RMSE: {np.mean(runs_rmse_array):0.6f} +/- {np.std(runs_rmse_array):0.6f}')

final_runtime = time.time() - runtime


print(f'TENDL-2021 R2: {np.mean(tendlr2s):0.3f} +/- {np.std(tendlr2s):0.3f}')
print(f'JENDL-5 R2: {np.mean(jendlr2s):0.3f} +/- {np.std(jendlr2s):0.3f}')
print(f'CENDL-3.2 R2: {np.mean(cendlr2s):0.3f} +/- {np.std(cendlr2s):0.3f}')
print(f'JEFF-3.3 R2: {np.mean(jeffr2s):0.3f} +/- {np.std(jeffr2s):0.3f}')
print(f'ENDF/B-VIII R2: {np.mean(endfbr2s):0.3f} +/- {np.std(endfbr2s):0.3f}')
print()
print(f"Runtime: {timedelta(seconds=final_runtime)}")
