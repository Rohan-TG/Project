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
import tqdm
# import shap
from sklearn.metrics import mean_squared_error, r2_score
from matrix_functions import range_setter, r2_standardiser, exclusion_func, General_plotter
from propagator_functions import make_train_sampler, make_test_sampler
import scipy.interpolate

df = pd.read_csv("ENDFBVIII_100keV_all_uncertainties.csv")


df.index = range(len(df))
al = range_setter(df=df, la=30, ua=208)

TENDL = pd.read_csv("TENDL_2021_MT_16_all_u.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=208)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=208)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=208)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=208)

n_run_tally95 = []

exc = exclusion_func()

validation_set_size = 20  # number of nuclides hidden from training

num_runs = 10
run_r2 = []
run_mse = []

nuclide_r2 = []
nuclide_thresholds = []

min95_consensus = []


anyotherbetter = []


nodd_peven = []
nodd_podd = []
neven_peven = []
neven_podd = []

atleast190 = []
atleast193 = []
atleast195 = []

min100A_at_least_95 = []
min100A_at_least_90 = []

olbjendl = []
olbtendl = []
olbcendl = []
olbjeff = []

eolbjendl = []
eolbcendl = []
eolbtendl = []
eolbjeff = []
eolbany = []
for q in tqdm.tqdm(range(num_runs)):
	nuclides_used = []
	every_prediction_list = []

	anyothertally = 0
	every_true_value_list = []

	endfb_r2s = []
	cendl_r2s = []
	tendl_r2s = []
	jeff_r2s = []
	jendl_r2s = []

	outliers = 0
	outliers90 = 0
	outliers85 = 0
	outliers9090 = 0
	outlier_tally = 0
	tally = 0

	tally_min100A_at_least_95 = 0
	tally_min100A_at_least_90 = 0


	exclusions_other_lib_better_jendl = 0
	exclusions_other_lib_better_cendl = 0
	exclusions_other_lib_better_tendl = 0
	exclusions_other_lib_better_jeff = 0
	exclusionsotherlibbetterany = 0

	tendlgeneral_olb = 0
	jendlgeneral_olb = 0
	cendlgeneral_olb = 0
	jeffgeneral_olb = 0

	low_masses = []

	nuclide_mse = []

	tally90 = 0

	other = []
	gewd_97 = 0

	bad_nuclides = []

	at_least_one_agreeing_90 = 0
	at_least_one_agreeing_93 = 0
	at_least_one_agreeing_95 = 0

	potential_outliers = []

	benchmark_total_library_evaluations = []
	benchmark_total_predictions = []

	low_mass_r2 = []
	while len(nuclides_used) < len(al):

		local_95 = 0

		# print(f"{len(nuclides_used) // len(al)} Epochs left")

		validation_nuclides = []  # list of nuclides used for validation


		while len(validation_nuclides) < validation_set_size:

			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
			if len(nuclides_used) == len(al):
				break


		# print("Test nuclide selection complete")
		print(f"{len(nuclides_used)}/{len(al)} selections")
		# print(f"Epoch {len(al) // len(nuclides_used) + 1}/")


		X_train, y_train = make_train_sampler(df=df, validation_nuclides=validation_nuclides, use_tqdm=False,
											  la=30, ua=208, exclusions=exc) # make training matrix

		X_test, y_test = make_test_sampler(validation_nuclides, df=df, use_tqdm=False)


		modelseed= random.randint(a=1, b=1000)
		model = xg.XGBRegressor(n_estimators=950,
								learning_rate=0.008,
								max_depth=8,
								subsample=0.18236,
								reg_lambda=0,
								max_leaves=0,
								seed=modelseed,)



		time1 = time.time()
		model.fit(X_train, y_train)

		# print("Training complete")

		predictions_ReLU = []

		for n in validation_nuclides:

			temp_x, temp_y = make_test_sampler(nuclides=[n], df=df, use_tqdm=False)
			initial_predictions = model.predict(temp_x)

			for p in initial_predictions:
				if p >= (0.02 * max(initial_predictions)):
					predictions_ReLU.append(p)
				else:
					predictions_ReLU.append(0.0)


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

			threshold_values = []
			for xs, energy in zip(pred_xs, erg):
				if xs > 0:
					predicted_threshold = energy
					break

			jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[nuc])
			cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[nuc])
			jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[nuc])
			tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[nuc])
			endfberg, endfbxs = General_plotter(df=df, nuclides=[nuc])

			interpolation_function = scipy.interpolate.interp1d(endfberg, y=pred_xs,
																fill_value='extrapolate')
			current_nuclide = nuc

			evaluation_r2s = []
			otherlibr2s = []
			eotherlibr2s = []

			truncated_library_r2 = []

			nuc_all_library_evaluations = []
			nuc_all_predictions = []


			for o, p in zip(endfberg, endfbxs):
				if p > 0:
					threshold_values.append(o)
					break

			try:
				pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(predicted_xs=pred_xs,
																				   library_xs=true_xs)
			except:
				print(current_nuclide)
			for libxs, p in zip(truncated_endfb, pred_endfb_gated):
				nuc_all_library_evaluations.append(libxs)
				nuc_all_predictions.append(p)

				benchmark_total_library_evaluations.append(libxs)
				benchmark_total_predictions.append(p)

			evaluation_r2s.append(endfb_r2)

			if current_nuclide in CENDL_nuclides:

				cendlxs_interpolated = interpolation_function(cendlerg)

				predcendlgated, truncatedcendl, cendl_r2 = r2_standardiser(library_xs=cendlxs,
																		   predicted_xs=cendlxs_interpolated)

				cendl_r2s.append(cendl_r2)
				otherlibr2s.append(cendl_r2)
				evaluation_r2s.append(cendl_r2)

				if cendl_r2 > endfb_r2:
					cendlgeneral_olb += 1

				for o, p in zip(cendlerg, cendlxs):
					if p > 0:
						threshold_values.append(o)
						break

				for x, y in zip(truncatedcendl, predcendlgated):
					benchmark_total_library_evaluations.append(x)
					benchmark_total_predictions.append(y)

					nuc_all_library_evaluations.append(x)
					nuc_all_predictions.append(y)

				# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

			if current_nuclide in JENDL_nuclides:
				jendlxs_interpolated = interpolation_function(jendlerg)

				predjendlgated, d2, jendl_r2 = r2_standardiser(library_xs=jendlxs, predicted_xs=jendlxs_interpolated)
				jendl_r2s.append(jendl_r2)
				evaluation_r2s.append(jendl_r2)
				otherlibr2s.append(jendl_r2)
				if jendl_r2 > endfb_r2:
					jendlgeneral_olb += 1

				for o, p in zip(jendlerg, jendlxs):
					if p > 0:
						threshold_values.append(o)
						break

				for x, y in zip(d2, predjendlgated):
					benchmark_total_library_evaluations.append(x)
					benchmark_total_predictions.append(y)

					nuc_all_library_evaluations.append(x)
					nuc_all_predictions.append(y)
				# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

			if current_nuclide in JEFF_nuclides:
				jeffxs_interpolated = interpolation_function(jefferg)

				predjeffgated, d2, jeff_r2 = r2_standardiser(library_xs=jeffxs,
															 predicted_xs=jeffxs_interpolated)

				if jeff_r2 > endfb_r2:
					jeffgeneral_olb += 1

				otherlibr2s.append(jeff_r2)

				for o, p in zip(jefferg, jeffxs):
					if p > 0:
						threshold_values.append(o)
						break
				jeff_r2s.append(jeff_r2)
				evaluation_r2s.append(jeff_r2)
				for x, y in zip(d2, predjeffgated):
					benchmark_total_library_evaluations.append(x)
					benchmark_total_predictions.append(y)

					nuc_all_library_evaluations.append(x)
					nuc_all_predictions.append(y)

			tendlxs_interpolated = interpolation_function(tendlerg)
			predtendlgated, truncatedtendl, tendlr2 = r2_standardiser(library_xs=tendlxs,
																	  predicted_xs=tendlxs_interpolated)
			for o, p in zip(tendlerg, tendlxs):
				if p > 0:
					threshold_values.append(o)
					break
			tendl_r2s.append(tendlr2)
			otherlibr2s.append(tendlr2)
			if tendlr2 > endfb_r2:
				tendlgeneral_olb += 1
			evaluation_r2s.append(tendlr2)
			for x, y in zip(truncatedtendl, predtendlgated):
				nuc_all_library_evaluations.append(x)
				nuc_all_predictions.append(y)

				benchmark_total_library_evaluations.append(x)
				benchmark_total_predictions.append(y)
				# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}"


			r2 = r2_score(nuc_all_library_evaluations,nuc_all_predictions) # consensus for the current nuclide

			if current_nuclide in exc:
				if current_nuclide in JENDL_nuclides:
					eotherlibr2s.append(jendl_r2)
					if jendl_r2 > endfb_r2:
						exclusions_other_lib_better_jendl += 1

				if current_nuclide in CENDL_nuclides:
					eotherlibr2s.append(cendl_r2)
					if cendl_r2 > endfb_r2:
						exclusions_other_lib_better_cendl += 1

				if current_nuclide in TENDL_nuclides:
					eotherlibr2s.append(tendlr2)
					if tendlr2 > endfb_r2:
						exclusions_other_lib_better_tendl += 1

				if current_nuclide in JEFF_nuclides:
					eotherlibr2s.append(jeff_r2)
					if jeff_r2 > endfb_r2:
						exclusions_other_lib_better_jeff += 1

			# print(f"{periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}: {r2}")

			thr_diffs = []
			for lib_thr in threshold_values:
				difference = lib_thr - predicted_threshold
				thr_diffs.append(difference)
			mean_difference = np.mean(thr_diffs)

			nuclide_thresholds.append([nuc[0], nuc[1], mean_difference])

			for jkl  in eotherlibr2s:
				if jkl > endfb_r2:
					exclusionsotherlibbetterany += 1
					break

			for jjj in otherlibr2s:
				if jjj >endfb_r2:
					anyothertally += 1
					break

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




			for z in evaluation_r2s:
				if z >= 0.9:
					at_least_one_agreeing_90 += 1
					break

			for u in evaluation_r2s:
				if nuc[1] >= 100 and u >= 0.95:
					tally_min100A_at_least_95 += 1
					break

			for q in evaluation_r2s:
				if nuc[1] >= 100 and q >= 0.90:
					tally_min100A_at_least_90 += 1
					break

			for z in evaluation_r2s:
				if z >=0.93:
					at_least_one_agreeing_93 += 1
					break

			for z in evaluation_r2s:
				if z >= 0.95:
					at_least_one_agreeing_95 += 1
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

		for n in nuclide_r2:
			if n[-1] >=0.95:
				local_95 += 1


		for pred in predictions:
			every_prediction_list.append(pred)

		for val in y_test:
			every_true_value_list.append(val)
		# overall_r2_list.append(overall_r2)
		# print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
		# print(f"R2: {overall_r2}") # Total R^2 for all predictions in this training campaign
		time_taken = time.time() - time1
		# print(f'completed in {time_taken:0.1f} s.\n')

	eolbjeff.append(exclusions_other_lib_better_jeff)
	eolbcendl.append(exclusions_other_lib_better_cendl)
	eolbjendl.append(exclusions_other_lib_better_jendl)
	eolbtendl.append(exclusions_other_lib_better_tendl)
	eolbany.append(exclusionsotherlibbetterany)

	anyotherbetter.append(anyothertally)

	olbtendl.append(tendlgeneral_olb)
	olbjeff.append(jeffgeneral_olb)
	olbcendl.append(cendlgeneral_olb)
	olbjendl.append(jendlgeneral_olb)
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

	atleast190.append(at_least_one_agreeing_90)
	atleast193.append(at_least_one_agreeing_93)
	atleast195.append(at_least_one_agreeing_95)

	min100A_at_least_90.append(tally_min100A_at_least_90)
	min100A_at_least_95.append(tally_min100A_at_least_95)

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
	time.sleep(2)
# print(f"At least one library r2 > 0.90: {at_least_one_agreeing_90}/{len(al)}")
# print(f"New overall r2: {r2_score(every_true_value_list, every_prediction_list)}")

# all_libraries_mse = mean_squared_error(y_true=all_library_evaluations, y_pred=all_predictions)
	benchmark_r2 = r2_score(y_true=benchmark_total_library_evaluations, y_pred= benchmark_total_predictions)
	benchmark_mse = mean_squared_error(y_true=benchmark_total_library_evaluations, y_pred=benchmark_total_predictions)
# print(f"MSE: {all_libraries_mse:0.5f}")
	print(f"R2: {benchmark_r2:0.5f}")
	run_r2.append(benchmark_r2) # stores all the benchmark r2s
	run_mse.append(benchmark_mse)

	print(f"Bad nuclides: {bad_nuclides}")

	min95_consensus.append(local_95)


agg_n_r2 = range_setter(df=df, la=30,ua=208)

alist = []

matrix95 = [[] for x in range(num_runs)]
for match in agg_n_r2:
	r2values = []
	for set in nuclide_r2:
		if [set[0], set[1]] == match:
			r2values.append(set[2])

	for j, run in enumerate(r2values):
		matrix95[j].append(run)


	alist.append([match[0], match[1], np.mean(r2values)])

A_plots = [i[1] for i in alist]
print(np.mean(run_r2))
print(np.std(run_r2))
print()












log_plots = [abs(np.log(abs(i[-1]))) for i in alist]
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
plt.ylabel("$|\ln(|r^2|)|$")
plt.title("Performance - A")
plt.grid()
plt.show()
#
# plt.figure()
# plt.plot(Z_plots, log_plots_Z, 'x')
# plt.xlabel('Z')
# plt.ylabel("$|\ln(|r^2|)|$")
# plt.title("Performance - Z")
# plt.grid()
# plt.show()

print(f'MSE: {np.mean(run_mse):0.6f} +/- {np.std(run_mse):0.6f}')

# plt.figure()
# plt.hist(x=log_plots, bins=50)
# plt.grid()
# plt.show()


plots = []
for nucleus in al:
	iteration_difference = []

	for set in nuclide_thresholds:
		if [set[0], set[1]] == nucleus:
			iteration_difference.append(set[-1])

	mean_n_difference = np.mean(iteration_difference)

	plots.append([nucleus[0], nucleus[1], mean_n_difference])


aplots = [a[1] for a in plots]
d = [x[-1] for x in plots]
plt.figure()
plt.plot(aplots, d, 'x')
plt.grid()
plt.ylabel('$\Delta$Threshold / MeV')
plt.xlabel('A')
plt.show()



plt.figure()
plt.hist(d,bins=40)
plt.grid()
plt.xlabel('Threshold difference / MeV')
plt.ylabel('Frequency')
plt.title('Mean difference between predicted threshold and evaluated thresholds')
plt.show()


print(f'At least one library r2 >=0.9: {np.mean(atleast190)} +- {np.std(atleast190)}')
print(f'At least one library r2 >=0.93: {np.mean(atleast193)} +- {np.std(atleast193)}')
print(f'At least one library r2 >=0.95: {np.mean(atleast195)} +- {np.std(atleast195)}')

print(f'Any other lib better than ENDFB: {np.mean(anyotherbetter)} +- {np.std(anyotherbetter)} out of {len(al)}')

print(f'Any other lib better for exclusions: {np.mean(eolbany)} +- {np.std(eolbany)} out of {len(exc)}')