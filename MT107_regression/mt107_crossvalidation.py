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
# import shap
from sklearn.metrics import mean_squared_error, r2_score
from functions107 import maketrain107, maketest107, Generalplotter107
from matrix_functions import r2_standardiser, range_setter

df = pd.read_csv('ENDFBVIII_MT107_nanremoved.csv')
al = range_setter(df=df, la=30, ua=208)

CENDL_32 = pd.read_csv('CENDL-3.2_MT107_main_features.csv')
CENDL_32= CENDL_32.dropna(subset=['XS'])
CENDL_nuclides = range_setter(df=CENDL_32, la=30, ua=260)
CENDL_32.index = range(len(CENDL_32))

JEFF_33 = pd.read_csv('JEFF33_all_features_MT16_103_107.csv')
JEFF_33 = JEFF_33.dropna(subset=['MT107XS'])
JEFF_nuclides = range_setter(df=JEFF_33, la=30, ua=260)
JEFF_33.index = range(len(JEFF_33))

JENDL_5 = pd.read_csv('JENDL-5_MT107_main_features.csv')
JENDL_nuclides = range_setter(df=JENDL_5, la=30, ua=260)
JENDL_5 = JENDL_5.dropna(subset=['XS'])
JENDL_5.index = range(len(JENDL_5))

TENDL_2021 = pd.read_csv('TENDL-2021_MT107_main_features.csv')
TENDL_2021 = TENDL_2021.dropna(subset=['XS'])
TENDL_nuclides = range_setter(df=TENDL_2021, la=30, ua=270)
TENDL_2021.index = range(len(TENDL_2021))

ENDFB_nuclides = range_setter(df=df, la=30, ua=260)
print("Data loaded...")



n_run_tally95 = []

goodnucs = [[14, 30],
 [15, 31],
 [16, 32],
 [16, 33],
 [16, 34],
 [16, 36],
 [17, 35],
 [17, 37],
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

# al = ENDFB_nuclides
validation_set_size = 20  # number of nuclides hidden from training

num_runs = 10
run_r2 = []

atleast90 = []
nuclide_r2 = []

for q in tqdm.tqdm(range(num_runs)):
	nuclides_used = []
	every_prediction_list = []
	every_true_value_list = []

	outliers = 0
	outliers90 = 0
	outliers85 = 0
	outliers9090 = 0
	outlier_tally = 0
	tally = 0


	atleast90tally = 0

	low_masses = []

	nuclide_mse = []

	tally90 = 0

	other = []
	gewd_97 = 0

	bad_nuclides = []

	at_least_one_agreeing_90 = 0

	every_prediction_list = []
	every_true_value_list = []
	potential_outliers = []

	benchmark_total_library_evaluations = []
	benchmark_total_predictions = []

	low_mass_r2 = []
	while len(nuclides_used) < len(al):

		# print(f"{len(nuclides_used) // len(al)} Epochs left")

		validation_nuclides = []  # list of nuclides used for validation


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


		X_train, y_train = maketrain107(df=df, validation_nuclides=validation_nuclides, la=30, ua=208) # make training matrix

		X_test, y_test = maketest107(validation_nuclides, df=df)

		# X_train must be in the shape (n_samples, n_features)
		# and y_train must be in the shape (n_samples) of the target

		print("Train/val matrices generated")

		modelseed= random.randint(a=1, b=1000)
		model = xg.XGBRegressor(n_estimators=900,
								learning_rate=0.01,
								max_depth=7,
								subsample=0.5,
								reg_lambda=2,
								seed = modelseed
								)

		time1 = time.time()
		model.fit(X_train, y_train)

		print("Training complete")

		predictions_ReLU = []

		for testnuclide in validation_nuclides:
			X_test_temp, y_test_temp = maketest107(nuclides=[testnuclide], df=df)

			temp_predictions = model.predict(X_test_temp)

			for p in temp_predictions:
				if p >= (0.02 * max(temp_predictions)):
					predictions_ReLU.append(p)
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



			try:
				pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(predicted_xs=pred_xs,
																				   library_xs=true_xs)
			except:
				print(current_nuclide)
			for libxs, p in zip(truncated_endfb, pred_endfb_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				benchmark_total_library_evaluations.append(libxs)
				benchmark_total_predictions.append(p)

			evaluation_r2s.append(endfb_r2)

			if current_nuclide in CENDL_nuclides:

				cendl_test, cendl_xs = maketest107(nuclides=[current_nuclide], df=CENDL_32)
				pred_cendl = model.predict(cendl_test)

				pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
				pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(predicted_xs=pred_cendl, library_xs=cendl_xs)
				for libxs, p in zip(truncated_cendl, pred_cendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)


					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)

				evaluation_r2s.append(pred_cendl_r2)
				truncated_library_r2.append(pred_cendl_r2)
				# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

			if current_nuclide in JENDL_nuclides:
				jendl_test, jendl_xs = maketest107(nuclides=[current_nuclide], df=JENDL_5)
				pred_jendl = model.predict(jendl_test)

				pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
				pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(predicted_xs=pred_jendl, library_xs=jendl_xs)
				for libxs, p in zip(truncated_jendl, pred_jendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)

					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)
				evaluation_r2s.append(pred_jendl_r2)
				truncated_library_r2.append(pred_jendl_r2)
				# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

			if current_nuclide in JEFF_nuclides:
				jeff_test, jeff_xs = maketest107(nuclides=[current_nuclide], df=JEFF_33)

				pred_jeff = model.predict(jeff_test)

				pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
				pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(predicted_xs=pred_jeff, library_xs=jeff_xs)
				for libxs, p in zip(truncated_jeff, pred_jeff_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)


					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)

				evaluation_r2s.append(pred_jeff_r2)
				truncated_library_r2.append(pred_jeff_r2)
				# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

			if current_nuclide in TENDL_nuclides:
				tendl_test, tendl_xs = maketest107(nuclides=[current_nuclide], df=TENDL_2021)
				pred_tendl = model.predict(tendl_test)

				pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
				pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(predicted_xs=pred_tendl, library_xs=tendl_xs)
				for libxs, p in zip(truncated_tendl, pred_tendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)


					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)

				evaluation_r2s.append(pred_tendl_r2)
				truncated_library_r2.append(pred_tendl_r2)
				# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}"


			r2 = r2_score(all_library_evaluations,all_predictions) # various comparisons

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

			for cc in evaluation_r2s:
				if cc > 0.90:
					atleast90tally += 1
					break



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
# plt.figure()
# plt.hist(x=low_masses)
# plt.show()

	low_a_badp = 0
	for a in bad_nuclides:
		if a[1] <=60:
			low_a_badp+=1
	print(f"{low_a_badp}/{len(bad_nuclides)} have A <= 60 and are bad performers")

	print(f"no. outliers estimate: {outliers}/{len(al)}")
	print()
	print(f"At least one library >= 0.95: {outlier_tally}/{len(al)}")
	n_run_tally95.append(outlier_tally)
	print(f"Estimate of outliers, threshold 0.9: {outliers90}/{len(al)}")
	print(f"outliers 90/90: {outliers9090}/{len(al)}")
	print(f"outliers 85/80: {outliers85}/{len(al)}")
	print(f"Tally of consensus r2 > 0.95: {tally}/{len(al)}")
	print(f"Tally of consensus r2 > 0.90: {tally90}/{len(al)}")
	# time.sleep(2)
# print(f"At least one library r2 > 0.90: {at_least_one_agreeing_90}/{len(al)}")
# print(f"New overall r2: {r2_score(every_true_value_list, every_prediction_list)}")

# all_libraries_mse = mean_squared_error(y_true=all_library_evaluations, y_pred=all_predictions)
	benchmark_r2 = r2_score(y_true=benchmark_total_library_evaluations, y_pred= benchmark_total_predictions)
# print(f"MSE: {all_libraries_mse:0.5f}")
	print(f"R2: {benchmark_r2:0.5f}")
	run_r2.append(benchmark_r2)
	atleast90.append(atleast90tally)

	print(f"Bad nuclides: {bad_nuclides}")

# print(f"Good predictions {tally}/{len(al)}")

agg_n_r2 = range_setter(df=df, la=0,ua=260)

alist = []
for match in agg_n_r2:
	r2values = []
	for set in nuclide_r2:
		if [set[0], set[1]] == match:
			r2values.append(set[2])

	alist.append([match[0], match[1], np.mean(r2values)])

A_plots = [i[1] for i in alist]
print(np.mean(run_r2))
print(np.std(run_r2))
# Z_plots = [i[0] for i in nuclide_r2]
# mse_log_plots = [np.log(i[-1]) for i in nuclide_mse]
# mse_plots = [i[-1] for i in nuclide_mse]

log_plots = [abs(np.log10(abs(i[-1]))) for i in alist]
# log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
#
# r2plot = [i[-1] for i in nuclide_r2]

# heavy_performance = [abs(np.log(abs(i[-1]))) for i in nuclide_r2 if i[1] < 50]
# print(f"Heavy performance: {heavy_performance}, mean: {np.mean(heavy_performance)}")
print(n_run_tally95)
plt.figure()
plt.plot(A_plots, log_plots, 'x')
# plt.plot(A_plots, r2plot, 'x')
plt.xlabel("A")
plt.ylabel("$|\lg(|r^2|)|$")
plt.title("Performance - A")
plt.grid()
plt.show()
#
# plt.figure()
# plt.plot(Z_plots, log_plots_Z, 'x')
# plt.xlabel('Z')
# plt.ylabel("$|\ln(|r^2|)|$")
# plt.title("Performance - Z")
print(f'At least 1 lib > 90: {np.mean(atleast90)} +- {np.std(atleast90)}')
# plt.grid()
# plt.show()

# plt.figure()
# plt.hist(x=log_plots, bins=50)
# plt.grid()
# plt.show()
