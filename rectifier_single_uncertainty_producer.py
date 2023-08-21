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
from matrix_functions import dsigma_dE, make_train, make_test,\
	range_setter, General_plotter
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df_test = df_test[df_test.Z != 11]
df = df[df.Z != 11]

df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))


al = range_setter(la=30, ua=215, df=df)




n_evaluations = 50
datapoint_matrix = []
target_nuclide = [40,95]

for i in tqdm.tqdm(range(n_evaluations)):
	print(f"\nRun {i+1}/{n_evaluations}")


	validation_nuclides = [target_nuclide]
	validation_set_size = 20  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("\nTest nuclide selection complete")

	time1 = time.time()

	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215,) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=df_test,)

	print("Data prep done")

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.008,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=model_seed,)


	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions
	predictions_ReLU = []
	for pred in predictions:
		if pred >= 0.003:
			predictions_ReLU.append(pred)
		else:
			predictions_ReLU.append(0)

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

	# plot predictions against data
	# note: python lists allow elements to be lists of varying lengths. This would not work using numpy arrays; the for
	# loop below loops through the lists ..._plotmatrix, where each element is a list corresponding to nuclide nuc[i].
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i] # validation nuclide
		r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide

	exp_true_xs = [y for y in y_test]
	exp_pred_xs = [xs for xs in predictions]

	print(f"MSE: {mean_squared_error(exp_true_xs, exp_pred_xs, squared=False)}") # MSE
	print(f"R2: {r2_score(exp_true_xs, exp_pred_xs):0.5f}") # Total R^2 for all predictions in this training campaign
	print(f'completed in {time.time() - time1:0.1f} s')


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



TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)
tendl_energy, tendl_xs = General_plotter(df=TENDL, nuclides=[target_nuclide])


JEFF33 = pd.read_csv('JEFF33_all_features.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=210)
JEFF_energy, JEFF_XS = General_plotter(df=JEFF33, nuclides=[target_nuclide])


JENDL5 = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL5_energy, JENDL5_XS = General_plotter(df=JENDL5, nuclides=[target_nuclide])
JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=210)


CENDL32 = pd.read_csv('CENDL32_all_features.csv')
CENDL32.index = range(len(CENDL32))
CENDL32_energy, CENDL33_XS = General_plotter(df=CENDL32, nuclides=[target_nuclide])
CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=210)
mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)
r2 = r2_score(true_xs, pred_xs)  # R^2 score for this specific nuclide
print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}, MSE: {mse:0.5f}")
time.sleep(0.8)

pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
pred_endfb_r2 = r2_score(y_true=true_xs, y_pred=pred_xs)
print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

if target_nuclide in CENDL_nuclides:
	cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL32)
	pred_cendl = model.predict(cendl_test)

	pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
	pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=pred_cendl)
	print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

if target_nuclide in JENDL_nuclides:
	jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL5)
	pred_jendl = model.predict(jendl_test)

	pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
	pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=pred_jendl)
	print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")

if target_nuclide in JEFF_nuclides:
	jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF33)

	pred_jeff = model.predict(jeff_test)

	pred_jeff_r2 = r2_score(y_true=jeff_xs, y_pred=pred_jeff)
	print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f}")

if target_nuclide in TENDL_nuclides:
	tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL)
	pred_tendl = model.predict(tendl_test)

	pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=pred_tendl)
	print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f}")



# Veeser_energies = [1.47000E+01,
# 1.60000E+01,
# 1.70000E+01,
# 1.80000E+01,
# 1.90000E+01,
# 2.00000E+01,]
# Veeser_delta_energies = [7.50000E-02,
# 1.00000E-01,
# 1.00000E-01,
# 7.50000E-02,
# 7.50000E-02,
# 5.50000E-02,]
#
# Veeser_XS = [1.98400E+03,
# 1.99200E+03,
# 1.78700E+03,
# 1.32000E+03,
# 9.57000E+02,
# 7.70000E+02,] # in mb
#
# Veeser_XS = [xs / 1000 for xs in Veeser_XS]
# Veeser_delta_XS = [1.15000E+02,
# 1.05000E+02,
# 8.50000E+01,
# 8.80000E+01,
# 6.40000E+01,
# 1.04000E+02,]
#
# Veeser_delta_XS = [delxs / 1000 for delxs in Veeser_delta_XS]
#
#
# Dzysiuk_energies = []
# Dzysiuk_delta_energies = []
#
# Dzysiuk_delta_XS = []
# Dzysiuk_delta_XS = [xs / 1000 for xs in Dzysiuk_delta_XS]
# Dzysiuk_XS = []
# Dzysiuk_XS = [xs / 1000 for xs in Dzysiuk_XS]
#
# Bayhurst_XS = [3.23000E+02,
# 8.16000E+02,
# 1.78900E+03,
# 1.66800E+03,
# 1.33700E+03,
# 9.65000E+02,
# 5.74000E+02,]
# Bayhurst_delta_XS = [1.50000E+01,
# 3.50000E+01,
# 7.60000E+01,
# 7.10000E+01,
# 0.00000E+00,
# 0.00000E+00,
# 0.00000E+00,]
# Bayhurst_XS = [xs/1000 for xs in Bayhurst_XS]
# Bayhurst_delta_XS = [xs/1000 for xs in Bayhurst_delta_XS]
#
#
# Bayhurst_energies = [8.51000E+00,
# 9.27000E+00,
# 1.41000E+01,
# 1.49200E+01,
# 1.71900E+01,
# 1.81900E+01,
# 1.99400E+01,]
# Bayhurst_delta_energies = [3.40000E-01,
# 3.60000E-01,
# 5.00000E-02,
# 5.00000E-02,
# 1.90000E-01,
# 1.30000E-01,
# 1.50000E-01,]
#
# Frehaut_E = [8.44000E+00,8.94000E+00,9.44000E+00,9.93000E+00,1.04200E+01,1.09100E+01,1.14000E+01,1.18800E+01,1.23600E+01,1.28500E+01,1.33300E+01,1.38000E+01,
# 1.42800E+01,1.47600E+01]
#
# Frehaut_E_d = [1.35000E-01,1.25000E-01,1.15000E-01,1.10000E-01,1.00000E-01,9.50000E-02,9.00000E-02,
# 8.50000E-02,8.50000E-02,8.00000E-02,7.50000E-02,7.50000E-02,7.00000E-02,6.50000E-02]
#
# Frehaut_XS = [2.03000E+02,6.59000E+02,1.16200E+03,1.40400E+03,1.56000E+03,1.72000E+03,1.69600E+03,1.81300E+03,1.91900E+03,1.99600E+03,2.03600E+03,
# 2.07600E+03,2.11300E+03,2.09400E+03]
#
# Frehaut_XS_d = [3.30000E+01 ,5.00000E+01 ,1.12000E+02 ,8.60000E+01 ,9.50000E+01 ,1.24000E+02 ,1.07000E+02 ,1.10000E+02 ,1.21000E+02 ,1.25000E+02 ,1.26000E+02 ,1.23000E+02 ,1.55000E+02 ,1.57000E+02 ]
#
# Frehaut_XS = [xs/1000 for xs in Frehaut_XS]
# Frehaut_XS_d = [xs/1000 for xs in Frehaut_XS_d]
#
#
#
#
#
#
#
#
#
#
#
#
# LuE = [1.44000E+01]
# LuXS = [ 1.440]
# LudXS = [0.08]
#
# QaimE = [1.47000E+01]
# QaimXS = [1.23000]
# QaimdXS = [0.146]
#
# JunhuaE = [1.35000E+01,1.41000E+01,1.48000E+01]
# JunhuaXS = [1.60000,1.73700,1.70700]
# JunhuadXS = [0.061, 0.07, 0.01]
#
# TemperleyE = [1.41000E+01]
# TemperleyXS = [1.57000]
# TemperleydXS = [0.18]
#
# BormannE = [1.26600E+01,1.33900E+01,1.40900E+01,1.50900E+01,1.61900E+01,1.69500E+01,1.75800E+01,1.81300E+01]
# BormannXS = [1.510,1.480,1.460,1.530,1.595,1.510,1.450,1.350]
# BormanndXS = [0.14,.140,.145,.135,.160,.135,.140,.130]
#
#
# #    E(MeV)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Frehaut_Pb_XS = [.8300,
# .1710,
# .3140,
# .6050,
# .9410,
# 1.184,
# 1.434,
# 1.580,
# 1.714,
# 1.804,
# 1.817,
# 1.921,
# 1.972,
# 2.071,
# 2.010,
# 2.028]
# Frehaut_Pb_E = [  7.93000E+00  ,
#   8.18000E+00  ,
#   8.44000E+00  ,
#   8.94000E+00  ,
#   9.44000E+00  ,
#   9.93000E+00  ,
#   1.04200E+01  ,
#   1.09100E+01  ,
#   1.14000E+01  ,
#   1.18800E+01  ,
#   1.23600E+01  ,
#   1.28500E+01  ,
#   1.33300E+01  ,
#   1.38000E+01  ,
#   1.42800E+01  ,
#   1.47600E+01  ]
# Frehaut_Pb_dXS = [0.0100,0.01400,0.02200,0.04000,.06100,.07500,.08300,.09700,.09800,.1050,.1370,.1430,.1500,.1600,
# .1580,
# .1630]
#
#
#
# FrehautW182_XS = [1.60000E+01,6.40000E+01,2.19000E+02,3.58000E+02,
# 7.21000E+02,1.14400E+03,1.36600E+03,1.58200E+03,
# 1.79000E+03,1.92300E+03,1.96000E+03,1.99000E+03,2.09200E+03,2.07600E+03]
# FrehautW182_XS = [i /1000 for i in FrehautW182_XS]
#
# FrehautW182_E = [8.18000E+00,8.44000E+00,8.69000E+00,8.94000E+00,9.44000E+00,
# 				 9.93000E+00,1.04200E+01,1.09100E+01,1.18800E+01,1.28500E+01,
# 				 1.33300E+01,1.38000E+01,1.42800E+01,1.47600E+01]
# FrehautW182_dXS = [1.20000E+01,1.20000E+01,2.80000E+01,2.20000E+01,4.10000E+01,6.00000E+01,7.00000E+01,8.00000E+01,
# 1.27000E+02,1.37000E+02,1.38000E+02,1.42000E+02,1.53000E+02,1.45000E+02]
# FrehautW182_dXS = [i / 1000 for i in FrehautW182_dXS]


#    E(MeV)       xs(mb)      dxs(mb)













# Frehautnd_XS = [2.26000E+02 ,5.59000E+02 ,
# 9.15000E+02 ,1.17300E+03 ,
# 1.30300E+03 ,1.46700E+03 ,
# 1.55600E+03 ,1.68800E+03 ,
# 1.84400E+03 ,1.94600E+03 ,
# 1.76200E+03 ]
# Frehautnd_XS = [i/1000 for i in Frehautnd_XS]
#
# Frehautnddxs = [1.80000E+01 ,3.10000E+01 ,
# 4.90000E+01 ,6.10000E+01 ,7.20000E+01 ,
# 7.60000E+01 ,8.20000E+01 ,1.20000E+02 ,
# 1.31000E+02 ,1.40000E+02 ,1.34000E+02 ]
# Frehautnddxs = [i/1000 for i in Frehautnddxs]
#
# Frehaut_E = [8.03000E+00,8.44000E+00,8.94000E+00,9.44000E+00,9.93000E+00,1.04200E+01,
# 1.09100E+01,1.18800E+01,1.28500E+01,1.38000E+01,1.47600E+01]
#
#
#
#
# anxs = [1.92300E+03,
# 1.76900E+03,
# 1.86700E+03,
# 1.67900E+03,
# 1.58500E+03,
# 1.42200E+03,
# 1.36900E+03,
# 1.34600E+03,
# 8.75000E+02,
# 7.62000E+02]
# andxs = [1.73000E+02,
# 1.41520E+02,
# 1.49360E+02,
# 1.34320E+02,
# 1.26800E+02,
# 1.13760E+02,
# 1.81000E+02,
# 1.40000E+02,
# 8.60000E+01,
# 7.90000E+01]
# andxs = [x / 1000 for x in andxs]
# anxs = [i/1000 for i in anxs]
# ane = [1.30000E+01 ,
# 1.34000E+01 ,
# 1.39000E+01 ,
# 1.45000E+01 ,
# 1.50000E+01 ,
# 1.54000E+01 ,
# 1.59000E+01 ,
# 1.66000E+01 ,
# 1.74000E+01 ,
# 1.79000E+01 ]
#
# KasugaiXS = [1.79000E+03,
# 1.79000E+03,
# 1.72000E+03,
# 1.69000E+03,
# 1.52000E+03,
# 1.43000E+03]
# KasugaiXS = [i/1000 for i in KasugaiXS]
#
# KasugaidXS = [1.20000E+02,
# 1.20000E+02,
# 1.10000E+02,
# 1.10000E+02,
# 1.00000E+02,
# 1.00000E+02]
# KasugaidXS = [i / 1000 for i in KasugaidXS]
# Kasugai_E = [1.33800E+01,
# 1.37000E+01,
# 1.40300E+01,
# 1.43600E+01,
# 1.46600E+01,
# 1.49500E+01]

# junhua_XS = [1.24600,1.27900,1.29100]
# junhua_XSd = [0.07,0.074,0.074]
# junhua_E = [1.35000E+01,1.41000E+01,1.48000E+01]


#    E(MeV)       xs(mb)      dxs(mb)













# meghaxs = [4.49100E+01,
# 2.14300E+02,
# 3.55810E+02,
# 5.28750E+02,
# 6.60350E+02,
# 8.00070E+02,
# 8.81830E+02,
# 9.41420E+02,
# 1.02667E+03,
# 1.07021E+03,
# 1.09026E+03]
# meghaxsd = [2.59000E+00,
# 1.22200E+01,
# 1.84600E+01,
# 3.02300E+01,
# 3.38200E+01,
# 5.93200E+01,
# 6.53800E+01,
# 4.50200E+01,
# 4.67800E+01,
# 5.31800E+01,
# 5.47100E+01]
# meghaxsd = [i/1000 for i in meghaxsd]
# meghaxs = [i/1000 for i in meghaxs]
# meghae = [9.90000E+00,
# 1.03900E+01,
# 1.08900E+01,
# 1.13900E+01,
# 1.18800E+01,
# 1.23800E+01,
# 1.28700E+01,
# 1.33700E+01,
# 1.38700E+01,
# 1.43600E+01,
# 1.48000E+01]
#
#
# puxs = [1.07900E+03,
# 1.21700E+03,
# 1.13900E+03]
# puxsd = [8.80000E+01,
# 1.02000E+02,
# 9.20000E+01]
# puxsd = [i/1000 for i in puxsd]
# puxs = [i/1000 for i in puxs]
# pue = [1.35000E+01,
# 1.41000E+01,
# 1.46000E+01]













# betake = [  1.47000E+01]
# betakxs = [0.613]
# betakxsd = [0.06]
#
# ikedae = [1.33300E+01,
# 1.35700E+01,
# 1.37500E+01,
# 1.39900E+01,
# 1.42200E+01,
# 1.44300E+01,
# 1.46700E+01,
# 1.49400E+01]
# ikedaxs = [6.77000E+02,
# 6.79000E+02,
# 7.11000E+02,
# 7.59000E+02,
# 7.45000E+02,
# 7.64000E+02,
# 8.05000E+02,
# 8.06000E+02]
# ikedaxs = [i/1000 for i in ikedaxs]
#
# ikedaxsd = [4.70000E+01,
# 5.60000E+01,
# 7.00000E+01,
# 5.00000E+01,
# 5.00000E+01,
# 4.80000E+01,
# 6.70000E+01,
# 7.80000E+01]
# ikedaxsd = [i/1000 for i in ikedaxsd]
#
# hillmane = [1.45000E+01]
# hillmanxs = [0.92]
# hillmanxsd = [0.184]
#
# tiwarie = [1.42000E+01]
# tiwarixs = [0.9]
# tiwarixsd = [0.108]
#
# cse = [13.28]
# csxs = [0.86]
# csxsd = [0.129]
#


print(f"Turning points: {dsigma_dE(XS=datapoint_means)}")

#2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendl_energy, tendl_xs, label = 'TENDL21', color='dimgrey')
if target_nuclide in JEFF_nuclides:
	plt.plot(JEFF_energy, JEFF_XS, '--', label='JEFF3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(JENDL5_energy, JENDL5_XS, label='JENDL5', color='green')
if target_nuclide in CENDL_nuclides:
	plt.plot(CENDL32_energy, CENDL33_XS, '--', label = 'CENDL3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
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
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.legend(loc='upper left')
plt.show()


final_runtime = time.time() - runtime

print(f"Runtime: {timedelta(seconds=final_runtime)}")
