import warnings

import periodictable
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
# import shap
from sklearn.metrics import mean_squared_error, r2_score
from matrix_functions import make_train, make_test, range_setter, r2_standardiser

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")  # dataframe as above, but with the new features from the Gilbert-Cameron model
al = range_setter(df=df, la=30, ua=210)

TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=260)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=260)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=260)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=260)

outliers = 0
outliers90 = 0
outliers85 = 0
outliers9090 = 0

low_masses = []

nuclides_used = []
nuclide_mse = []
nuclide_r2 = []
tally90 = 0

other = []
gewd_97 = 0

bad_nuclides = []

at_least_one_agreeing_90 = 0

every_prediction_list = []
every_true_value_list = []
potential_outliers = []

all_library_evaluations = []
all_predictions = []

low_mass_r2 = []

outlier_tally = 0

tally = 0
while len(nuclides_used) < len(al):


	# print(f"{len(nuclides_used) // len(al)} Epochs left")

	validation_nuclides = []  # list of nuclides used for validation
	# test_nuclides = []
	validation_set_size = 25  # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:

		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides and choice not in nuclides_used:
			validation_nuclides.append(choice)
			nuclides_used.append(choice)
		if len(nuclides_used) == len(al):
			break


	print("Test nuclide selection complete")
	print(f"{len(nuclides_used)}/{len(al)} selections")
	# print(f"Epoch {len(al) // len(nuclides_used) + 1}/")


	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=210) # make training matrix

	X_test, y_test = make_test(validation_nuclides, df=df_test)

	# X_train must be in the shape (n_samples, n_features)
	# and y_train must be in the shape (n_samples) of the target

	print("Train/val matrices generated")

	modelseed= random.randint(a=1, b=1000)
	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.008,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=modelseed,)

	# model = xg.XGBRegressor(n_estimators=600,
	# 						reg_lambda=1.959,
	# 						learning_rate=0.00856,
	# 						max_depth=7,
	# 						subsample=0.1895,
	# 						seed=42)

	time1 = time.time()
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
		nuc = validation_nuclides[i]
		current_nuclide = nuc

		evaluation_r2s = []

		truncated_library_r2 = []

		all_library_evaluations = []
		all_predictions = []

		limited_evaluations = []
		limited_predictions = []

		try:
			pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(raw_predictions=pred_xs,
																			   library_xs=true_xs)
		except:
			print(current_nuclide)
		for libxs, p in zip(truncated_endfb, pred_endfb_gated):
			all_library_evaluations.append(libxs)
			all_predictions.append(p)
		evaluation_r2s.append(endfb_r2)

		if current_nuclide in CENDL_nuclides:

			cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL)
			pred_cendl = model.predict(cendl_test)

			pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
			pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl, library_xs=cendl_xs)
			for libxs, p in zip(truncated_cendl, pred_cendl_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)
			evaluation_r2s.append(pred_cendl_r2)
			truncated_library_r2.append(pred_cendl_r2)
			# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

		if current_nuclide in JENDL_nuclides:
			jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL)
			pred_jendl = model.predict(jendl_test)

			pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
			pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl, library_xs=jendl_xs)
			for libxs, p in zip(truncated_jendl, pred_jendl_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)
			evaluation_r2s.append(pred_jendl_r2)
			truncated_library_r2.append(pred_jendl_r2)
			# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

		if current_nuclide in JEFF_nuclides:
			jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF)

			pred_jeff = model.predict(jeff_test)

			pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
			pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff, library_xs=jeff_xs)
			for libxs, p in zip(truncated_jeff, pred_jeff_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)

			evaluation_r2s.append(pred_jeff_r2)
			truncated_library_r2.append(pred_jeff_r2)
			# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

		if current_nuclide in TENDL_nuclides:
			tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL)
			pred_tendl = model.predict(tendl_test)

			pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
			pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl, library_xs=tendl_xs)
			for libxs, p in zip(truncated_tendl, pred_tendl_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				limited_evaluations.append(libxs)
				limited_predictions.append(p)

			evaluation_r2s.append(pred_tendl_r2)
			truncated_library_r2.append(pred_tendl_r2)
			# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}"


		r2 = r2_score(all_library_evaluations,all_predictions) # various comparisons

		limited_r2 = r2_score(limited_evaluations,limited_predictions)
		# print(f"{periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}: {r2}")

		if r2 > 0.97 and endfb_r2 < 0.9:
			outliers += 1

		if r2 > 0.95 and endfb_r2 <= 0.9:
			outliers90 += 1

		if r2 > 0.85 and endfb_r2 <0.8:
			outliers85 += 1

		if r2 >0.9 and endfb_r2 <0.9:
			outliers9090+=1

		for z in evaluation_r2s:
			if z > 0.95:
				outlier_tally += 1
				break


		if limited_r2 - endfb_r2 > 0.02:
			potential_outliers.append(current_nuclide)

		for z in evaluation_r2s:
			if z > 0.9:
				at_least_one_agreeing_90 += 1
				break

		l_temp = 0
		for l in evaluation_r2s:
			if l < 0.9:
				l_temp += 1
			if int(l_temp) == len(evaluation_r2s):
				bad_nuclides.append(current_nuclide)

		if nuc[1] <= 60:
			low_mass_r2.append(r2)

		if nuc[1] > 60:
			other.append(r2)

		if r2 > 0.95:
			tally += 1

		if r2 >= 0.9:
			tally90 += 1

		if r2 >= 0.97:
			gewd_97 +=1
		if current_nuclide[1] <= 60:
			low_masses.append(r2)
		# r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
		# print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")

		mse = mean_squared_error(true_xs, pred_xs)

		nuclide_mse.append([nuc[0], nuc[1], mse])
		nuclide_r2.append([nuc[0], nuc[1], r2])
		# individual_r2_list.append(r2)

	# overall_r2 = r2_score(y_test, predictions)
	for pred in predictions:
		every_prediction_list.append(pred)

	for val in y_test:
		every_true_value_list.append(val)
	# overall_r2_list.append(overall_r2)
	# print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
	# print(f"R2: {overall_r2}") # Total R^2 for all predictions in this training campaign
	time_taken = time.time() - time1
	print(f'completed in {time_taken:0.1f} s.\n')


lownucs = range_setter(df=df, la=0, ua=60)
othernucs = range_setter(df=df, la=61, ua=210)
oncount = 0
oncount90=0
for q in other:
	if q < 0.8:
		oncount+=1
	if q < 0.9:
		oncount90+=1

print(f"{oncount}/{len(othernucs)} of mid nucs have r2 below 0.8")
print(f"{oncount90}/{len(othernucs)} of mid nucs have r2 below 0.9")
print(f">= 97 consensus: {gewd_97}/{len(al)}")

lncount = 0
lncount90 = 0
for u in low_masses:
	if u < 0.8:
		lncount+=1
	if u < 0.9:
		lncount90+=1
print(f"{lncount}/{len(lownucs)} of A<=60 nuclides have r2 below 0.8")
print(f"{lncount90}/{len(lownucs)} of A<=60 nuclides have r2 below 0.9")
plt.figure()
plt.hist(x=low_masses)
plt.show()

low_a_badp = 0
for a in bad_nuclides:
	if a[1] <=60:
		low_a_badp+=1
print(f"{low_a_badp}/{len(bad_nuclides)} have A <= 60 and are bad performers")

print(f"no. outliers estimate: {outliers}/{len(al)}")
print()
print(f"At least one library >= 0.95: {outlier_tally}/{len(al)}")
print(f"Estimate of outliers, threshold 0.9: {outliers90}/{len(al)}")
print(f"outliers 90/90: {outliers9090}/{len(al)}")
print(f"outliers 85/80: {outliers85}/{len(al)}")
print(f"Tally of consensus r2 > 0.95: {tally}/{len(al)}")
print(f"Tally of consensus r2 > 0.90: {tally90}/{len(al)}")
# print(f"At least one library r2 > 0.90: {at_least_one_agreeing_90}/{len(al)}")
# print(f"New overall r2: {r2_score(every_true_value_list, every_prediction_list)}")

# all_libraries_mse = mean_squared_error(y_true=all_library_evaluations, y_pred=all_predictions)
all_libraries_r2 = r2_score(y_true=all_library_evaluations, y_pred= all_predictions)
# print(f"MSE: {all_libraries_mse:0.5f}")
print(f"R2: {all_libraries_r2:0.5f}")

print(f"Bad nuclides: {bad_nuclides}")

# print(f"Good predictions {tally}/{len(al)}")

A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]
# mse_log_plots = [np.log(i[-1]) for i in nuclide_mse]
# mse_plots = [i[-1] for i in nuclide_mse]

log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]

r2plot = [i[-1] for i in nuclide_r2]

# heavy_performance = [abs(np.log(abs(i[-1]))) for i in nuclide_r2 if i[1] < 50]
# print(f"Heavy performance: {heavy_performance}, mean: {np.mean(heavy_performance)}")

plt.figure()
plt.plot(A_plots, log_plots, 'x')
# plt.plot(A_plots, r2plot, 'x')
plt.xlabel("A")
plt.ylabel("$|\ln(|r^2|)|$")
plt.title("Performance - A")
plt.grid()
plt.show()

plt.figure()
plt.plot(Z_plots, log_plots_Z, 'x')
plt.xlabel('Z')
plt.ylabel("$|\ln(|r^2|)|$")
plt.title("Performance - Z")
plt.grid()
plt.show()

# plt.figure()
# plt.hist(x=log_plots, bins=50)
# plt.grid()
# plt.show()

plt.figure()
plt.hist2d(x=A_plots, y=log_plots, bins=20)
plt.xlabel("A")
plt.ylabel("$|\ln(|r^2|)|$")
plt.colorbar()
plt.grid()
plt.show()

# tally2 = 0
# for x in low_mass_r2:
# 	if x < 0.95:
# 		tally2 += 1


# lowmass = range_setter(df=df, la=30, ua=60)
# print(f"{tally2}/{len(lowmass)}")

print(potential_outliers)