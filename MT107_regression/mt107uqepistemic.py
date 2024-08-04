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

df = pd.read_csv("ENDFBVIII_MT107_nanremoved.csv")


df.index = range(len(df))


# al = range_setter(la=30, ua=215, df=df)
al = [[14, 30],
 [15, 31],
 [16, 32],
 [16, 33],
 [16, 34],
 [16, 36],
 [17, 35],
 # [17, 37],
 [18, 36],
 [18, 37],
 [18, 38],
 [18, 40],
 [18, 41],
 [19, 39],
 [19, 40],
 [19, 41],
 [20, 40],
 [20, 42],
 [20, 43],
 [20, 44],
 [20, 45],
 [20, 46],
 [20, 47],
 [20, 48],
 [21, 45],
 [22, 46],
 [22, 47],
 [22, 48],
 [22, 49],
 [22, 50],
 [23, 50],
 [23, 51],
 [24, 50],
 [24, 51],
 [24, 52],
 [24, 53],
 [24, 54],
 [25, 54],
 [26, 55],
 [27, 58],
 [27, 59],
 [28, 58],
 [28, 59],
 [28, 60],
 [28, 61],
 [28, 62],
 [28, 64],
 [29, 63],
 [29, 65],
 [30, 64],
 [30, 65],
 [30, 66],
 [30, 67],
 [30, 68],
 [30, 70],
 [31, 69],
 [31, 71],
 [32, 70],
 [32, 72],
 [32, 73],
 [32, 74],
 [32, 76],
 [33, 73],
 [33, 74],
 [33, 75],
 [34, 74],
 [34, 75],
 [34, 76],
 [34, 77],
 [34, 78],
 [34, 79],
 [34, 80],
 [34, 82],
 [35, 79],
 [35, 81],
 [36, 78],
 [36, 80],
 [36, 81],
 [36, 82],
 [36, 83],
 [36, 84],
 [36, 85],
 [36, 86],
 [37, 85],
 [37, 86],
 [37, 87],
 [38, 84],
 [38, 86],
 [38, 87],
 [38, 88],
 [38, 89],
 [38, 90],
 [39, 89],
 [39, 91],
 [41, 93],
 [41, 94],
 [41, 95],
 [42, 92],
 [42, 93],
 [42, 94],
 [42, 95],
 [42, 96],
 [42, 97],
 [42, 98],
 [42, 99],
 [42, 100],
 [43, 98],
 [43, 99],
 [44, 96],
 [44, 97],
 [44, 98],
 [44, 99],
 [44, 100],
 [44, 101],
 [44, 102],
 [44, 103],
 [44, 104],
 [44, 105],
 [44, 106],
 [45, 103],
 [45, 105],
 [46, 102],
 [46, 104],
 [46, 105],
 [46, 106],
 [46, 107],
 [46, 108],
 [46, 110],
 [47, 107],
 [47, 109],
 [47, 111],
 [48, 106],
 [48, 108],
 [48, 109],
 [48, 110],
 [48, 111],
 [48, 112],
 [48, 113],
 [48, 114],
 [48, 116],
 [49, 113],
 [49, 115],
 [50, 112],
 [50, 113],
 [50, 114],
 [50, 115],
 [50, 116],
 [50, 117],
 [50, 118],
 [50, 119],
 [50, 120],
 [50, 122],
 [50, 123],
 [50, 124],
 [50, 125],
 [50, 126],
 [51, 121],
 [51, 123],
 [51, 124],
 [51, 125],
 [51, 126],
 [52, 120],
 [52, 122],
 [52, 123],
 [52, 124],
 [52, 125],
 [52, 126],
 [52, 128],
 [52, 130],
 [52, 132],
 [53, 127],
 [53, 129],
 [53, 130],
 [53, 131],
 [53, 135],
 [54, 123],
 [54, 124],
 [54, 126],
 [54, 128],
 [54, 129],
 [54, 130],
 [54, 131],
 [54, 132],
 [54, 133],
 [54, 134],
 [54, 135],
 [54, 136],
 [55, 133],
 [55, 134],
 [55, 135],
 [55, 136],
 [55, 137],
 [56, 130],
 [56, 132],
 [56, 133],
 [56, 134],
 [56, 135],
 [56, 136],
 [56, 137],
 [56, 138],
 [56, 140],
 [57, 138],
 [57, 139],
 [57, 140],
 [58, 136],
 [58, 138],
 [58, 139],
 [58, 140],
 [58, 141],
 [58, 142],
 [58, 143],
 [58, 144],
 [59, 141],
 [59, 142],
 [59, 143],
 [60, 142],
 [60, 143],
 [60, 144],
 [60, 145],
 [60, 146],
 [60, 147],
 [60, 148],
 [60, 150],
 [61, 143],
 [61, 144],
 [61, 145],
 [61, 147],
 [61, 148],
 [61, 149],
 [61, 151],
 [62, 144],
 [62, 145],
 [62, 147],
 [62, 148],
 [62, 149],
 [62, 150],
 [62, 151],
 [62, 152],
 [62, 153],
 [62, 154],
 [63, 151],
 [63, 152],
 [63, 153],
 [63, 154],
 [63, 155],
 [63, 156],
 [63, 157],
 [64, 152],
 [64, 153],
 [64, 154],
 [64, 155],
 [64, 156],
 [64, 157],
 [64, 158],
 [64, 160],
 [65, 159],
 [65, 160],
 [66, 154],
 [66, 156],
 [66, 158],
 [66, 159],
 [66, 160],
 [66, 161],
 [66, 162],
 [66, 163],
 [66, 164],
 [68, 162],
 [68, 164],
 [68, 166],
 [68, 167],
 [68, 168],
 [68, 170],
 [70, 168],
 [70, 170],
 [70, 171],
 [70, 172],
 [70, 173],
 [70, 174],
 [70, 176],
 [71, 175],
 [71, 176],
 [72, 174],
 [72, 176],
 [72, 177],
 [72, 178],
 [72, 179],
 [72, 180],
 [72, 181],
 [72, 182],
 [73, 180],
 [73, 181],
 [73, 182],
 [75, 185],
 [75, 187],
 [76, 184],
 [76, 186],
 [76, 187],
 [76, 188],
 [76, 189],
 [76, 190],
 [76, 191],
 [76, 192],
 [77, 191],
 [77, 192],
 [77, 193],
 [78, 190],
 [78, 191],
 [78, 192],
 [78, 193],
 [78, 194],
 [78, 195],
 [78, 196],
 [78, 197],
 [78, 198],
 [79, 197],
 [80, 196],
 [80, 198],
 [80, 199],
 [80, 200],
 [80, 201],
 [80, 202],
 [80, 203],
 [80, 204],
 [81, 204],
 [82, 204],
 [82, 205],
 [82, 206],
 [82, 207],
 [82, 208],
 [83, 209],
 [84, 208],
 [84, 210]]

TENDL = pd.read_csv("TENDL-2021_MT107_main_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)


JEFF33 = pd.read_csv('JEFF33_all_features_MT16_103_107.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=210)


JENDL5 = pd.read_csv('JENDL-5_MT107_main_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=210)


CENDL32 = pd.read_csv('CENDL-3.2_MT107_main_features.csv')
CENDL32.index = range(len(CENDL32))
CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=210)


n_evaluations = 4
datapoint_matrix = []
target_nuclide = [60,148]

jendlerg, jendlxs = Generalplotter107(JENDL5, target_nuclide)
cendlerg, cendlxs = Generalplotter107(CENDL32,target_nuclide)
jefferg, jeffxs = Generalplotter107(JEFF33, target_nuclide)
tendlerg, tendlxs = Generalplotter107(TENDL, target_nuclide)
endfberg, endfbxs = Generalplotter107(df, target_nuclide)

runs_r2_array = []
runs_rmse_array = []

X_test, y_test = maketest107([target_nuclide], df=df, )

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
								  la=20, ua=208, )  # make training matrix


	# print("Training...")

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=600,
							learning_rate=0.02,
							max_depth=7,
							subsample=0.888,
							reg_lambda=1,
							seed=model_seed
							)

	model.fit(X_train, y_train)

	# print("Training complete")

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


	pred_endfb_r2 = r2_score(y_true=XS_plotmatrix[0], y_pred=P_plotmatrix[0])
	for x, y in zip(XS_plotmatrix[0], P_plotmatrix[0]):
		all_libs.append(y)
		all_preds.append(x)
	# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

	if target_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = maketest107(nuclides=[target_nuclide], df=CENDL32)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=pred_cendl)
		for x, y in zip(pred_cendl, cendl_xs):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - CENDL-3.2 R2: {pred_cendl_r2:0.5f}")

	if target_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = maketest107(nuclides=[target_nuclide], df=JENDL5)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=pred_jendl)
		# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")
		for x, y in zip(pred_jendl, jendl_xs):
			all_libs.append(y)
			all_preds.append(x)

	if target_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = maketest107(nuclides=[target_nuclide], df=JEFF33)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(pred_jeff, jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - JEFF-3.3 R2: {pred_jeff_r2:0.5f}")

	if target_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = maketest107(nuclides=[target_nuclide], df=TENDL)
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










#2sigma CF
plt.figure()
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendlerg, tendlxs, label = 'TENDL-2021', color='dimgrey')
if target_nuclide in JEFF_nuclides:
	plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
if target_nuclide in CENDL_nuclides:
	plt.plot(cendlerg, cendlxs, '--', label = 'CENDL-3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
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
