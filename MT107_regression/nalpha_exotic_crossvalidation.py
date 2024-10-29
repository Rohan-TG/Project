import warnings

import periodictable
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
# import time
import tqdm
# import shap
from sklearn.metrics import mean_squared_error, r2_score
from matrix_functions import range_setter
from functions107 import maketest107, maketrain107, Generalplotter107
import scipy.interpolate

df = pd.read_csv("ENDFBVIII_MT_103_all_features.csv")

gate = 0.0

df.index = range(len(df))
df = df[(df['Z'] != 6) & (df['A'] != 12)]
df.index = range(len(df))
al = range_setter(df=df, la=0, ua=208)

TENDL = pd.read_csv("TENDL-2021_MT_107_all_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=208)

JEFF = pd.read_csv('JEFF-3.3_MT_107_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=208)

JENDL = pd.read_csv('JENDL-5_MT_107_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=208)

CENDL = pd.read_csv('CENDL-3.2_MT_107_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=208)


num_runs = 2

run_r2 = []
run_mse = []

nuclide_r2 = []
nuclide_thresholds = []
minenergy = 0.1
maxenergy = 20

def diff(target):
	match_element = []
	zl = [n[0] for n in al]
	if target[0] in zl:
		for nuclide in al:
			if nuclide[0] == target[0]:
				match_element.append(nuclide)
		nuc_differences = []
		for match_nuclide in match_element:
			difference = match_nuclide[1] - target[1]
			nuc_differences.append(difference)
		if len(nuc_differences) > 0:
			if nuc_differences[0] > 0:
				min_difference = min(nuc_differences)
			elif nuc_differences[0] < 0:
				min_difference = max(nuc_differences)
		return (-1 * min_difference)



lmdcount = []

hmdcount = []

validation_nuclides = []
for nuc in TENDL_nuclides:
	if nuc not in al:
		validation_nuclides.append(nuc)


for q in tqdm.tqdm(range(num_runs)):
	nuclides_used = []
	every_prediction_list = []
	every_true_value_list = []
	mdcount80 = 0
	mdcount90 = 0

	highmd80 = 0
	highmd90 = 0

	high_md = []
	tendl_r2s = []
	low_md = []
	jeff_r2s = []


	tendl_rmses = []
	jendl_r2s = []

	outliers = 0
	outlier_tally = 0
	tally = 0

	low_masses = []

	nuclide_mse = []

	tally90 = 0

	other = []

	bad_nuclides = []


	potential_outliers = []

	benchmark_total_library_evaluations = []
	benchmark_total_predictions = []
	# print('Forming data...')

	X_train, y_train = maketrain107(df=df, validation_nuclides=validation_nuclides, la=0, ua=208,
							minerg=minenergy, maxerg=maxenergy)  # make training matrix

	X_test, y_test = maketest107(validation_nuclides, df=TENDL, minerg=minenergy, maxerg=maxenergy)

	modelseed= random.randint(a=1, b=1000)
	print('Training...')
	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.01,
							max_depth=7,
							subsample=0.2,
                            reg_lambda=2,
							max_leaves=0,
							seed=modelseed,)


	model.fit(X_train, y_train)

	predictions = model.predict(X_test)

	print('Inference complete')

	XS_plotmatrix = []
	E_plotmatrix = []
	P_plotmatrix = []
	for nuclide in tqdm.tqdm(validation_nuclides, total=len(validation_nuclides)):
		dummy_test_XS = []
		dummy_test_E = []
		dummy_predictions = []
		for i, row in enumerate(X_test):
			if [row[0], row[1]] == nuclide:
				dummy_test_XS.append(y_test[i])
				dummy_test_E.append(row[4])  # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	for i, (pred_xs, true_xs, erg) in tqdm.tqdm(enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)), total = len(P_plotmatrix)):
		nuc = validation_nuclides[i]

		evaluation_r2s = []
		gated_p = []
		for p in pred_xs:
			if p >= (gate * max(pred_xs)):
				gated_p.append(p)
			else:
				gated_p.append(0.0)

		pred_xs = gated_p


		nuc_all_library_evaluations = []
		nuc_all_predictions = []

		jendlerg, jendlxs = Generalplotter107(dataframe=JENDL, nuclide=nuc, maxenergy=maxenergy, minenergy=minenergy)

		interpolation_function = scipy.interpolate.interp1d(erg, y=pred_xs,
															fill_value='extrapolate')

		tendl_r2 = r2_score(true_xs, pred_xs)
		tendl_rmse = mean_squared_error(true_xs, pred_xs)**0.5
		for libxs, p in zip(true_xs, pred_xs):
			nuc_all_library_evaluations.append(libxs)
			nuc_all_predictions.append(p)

			benchmark_total_library_evaluations.append(libxs)
			benchmark_total_predictions.append(p)

		evaluation_r2s.append(tendl_r2)

		current_nuclide = nuc

		truncated_library_r2 = []



		if current_nuclide in JENDL_nuclides:
			jendlxs_interpolated = interpolation_function(jendlerg)

			jendl_r2 = r2_score(jendlxs, jendlxs_interpolated)
			jendl_r2s.append(jendl_r2)


			for x, y in zip(jendlxs, jendlxs_interpolated):
				benchmark_total_library_evaluations.append(x)
				benchmark_total_predictions.append(y)

				nuc_all_library_evaluations.append(x)
				nuc_all_predictions.append(y)

			evaluation_r2s.append(jendl_r2)

		r2 = r2_score(nuc_all_library_evaluations, nuc_all_predictions)

		rmsescore = mean_squared_error(nuc_all_library_evaluations, nuc_all_predictions) **0.5

		nuclide_r2.append([nuc[0], nuc[1], r2, rmsescore])

		md = diff(nuc)
		try:
			if abs(md) <= 3:
				low_md.append(nuc)
				if r2 >= 0.8:
					mdcount80 += 1
				if r2 >= 0.9:
					mdcount90 += 1


			if abs(md) > 3:
				high_md.append(nuc)
				if r2>= 0.8:
					highmd80 += 1
				if r2>=90:
					highmd90 += 1
		except TypeError:
			print('No match for ', nuc)

	lmdcount.append(mdcount90)
	hmdcount.append(highmd90)

zl = [n[0] for n in al]
print(len(zl))

mins = []
for c in validation_nuclides:
	if c[0] in zl:
		mins.append(c)

print(len(mins))

print('r2 > 80:',mdcount80 , '/', len(low_md))
print('r2 > 90:',mdcount90 , '/', len(low_md))

print('r2 > 80:', highmd80, '/', len(high_md))
print('r2 > 90:', highmd90, '/', len(high_md))

agg_n_r2 = range_setter(df=TENDL, la=30,ua=208)

exotics = []
for ll in agg_n_r2:
	if ll not in al:
		exotics.append(ll)

alist = []
for match in exotics:
	r2values = []
	rmsevalues = []
	for set in nuclide_r2:
		if [set[0], set[1]] == match:
			r2values.append(set[2])
			rmsevalues.append(set[3])


	difference = diff([match[0], match[1]])

	alist.append([match[0], match[1], np.mean(r2values), np.mean(rmsevalues), np.std(r2values),
				  difference])

A_plots = [i[1] for i in alist]
log_plots = [abs(np.log(abs(i[-3]))) for i in alist]



plt.figure()
plt.plot(A_plots, log_plots, 'x')
# plt.plot(A_plots, r2plot, 'x')
plt.xlabel("A")
plt.ylabel("$|\ln(|r^2|)|$")
plt.title("Performance - A")
plt.grid()
plt.show()