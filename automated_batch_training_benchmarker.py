import pandas as pd
import matplotlib.pyplot as plt
import periodictable
import scipy
import tqdm

from matrix_functions import General_plotter, range_setter, make_train, make_test, dsigma_dE, r2_standardiser
import random
import shap
import numpy as np
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error
import time

TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=210)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=210)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=210)

df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

df = df[df.Z != 6]
df_test = df_test[df_test.Z != 6]

df_test.index = range(len(df_test)) # re-label indices
df = df[df.A <= 210]
df.index = range(len(df))
kx = range_setter(la=30, ua=210, df=df)


banned_nuclides = [[3, 7],
[4, 9],
[1, 2],
[1, 3],
[7, 14]]



exc = [[22, 47], [65, 159], [66, 157], [38, 90], [61, 150],
	   [74, 185], [50, 125], [50, 124], [60, 149], [39, 90],
	   [64, 160], [38, 87], [39, 91], [63, 152], [52, 125],
	   [19, 40], [56, 139], [52, 126], [71, 175], [34, 79],
	   [70, 175], [50, 117], [23, 49], [63, 156], [57, 140],
	   [52, 128], [59, 142], [50, 118], [50, 123], [65, 161],
	   [52, 124], [38, 85], [51, 122], [19, 41], [54, 135],
	   [32, 75], [81, 205], [71, 176], [72, 175], [50, 122],
	   [51, 125], [53, 133], [34, 82], [41, 95], [46, 109],
	   [84, 209], [56, 140], [64, 159], [68, 167], [16, 35],
	   [18,38]] # 10 sigma with handpicked additions


al = []
for i in kx:
	if i not in banned_nuclides:
		al.append(i)

validation_set_size = 15



n_runs = 10
lower_bound = 30
upper_bound = 79
run_r2 = []

#

for i in tqdm.tqdm(range(n_runs+1)):
	LA = lower_bound
	UA = upper_bound
	every_prediction_list = []
	every_true_value_list = []
	benchmark_nuclides_used = []
	nuclide_r2 = []

	while len(benchmark_nuclides_used) < len(al):
		time1 = time.time()





		nuclide_subset = range_setter(df=df, la=LA, ua=UA) # generate new subset for stratum
		group_nuclides_used = []


		while len(group_nuclides_used) < len(nuclide_subset):
			validation_nuclides = []
			while len(validation_nuclides) < validation_set_size:

				if len(group_nuclides_used) == len(nuclide_subset):
					break

				validation_nuclide = random.choice(nuclide_subset)
				if validation_nuclide not in benchmark_nuclides_used and validation_nuclide not in validation_nuclides:
					validation_nuclides.append(validation_nuclide)
					group_nuclides_used.append(validation_nuclide)
					benchmark_nuclides_used.append(validation_nuclide)
				else:
					# print(validation_nuclides)
					continue

			print(validation_nuclides)

			print(f"{len(benchmark_nuclides_used)}/{len(al)} total selections")
			print(f"{len(group_nuclides_used)}/{len(nuclide_subset)} group selections")

			X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, ua=UA, la=LA, exclusions=exc)

			X_test, y_test = make_test(nuclides=validation_nuclides, df=df)

			print("Train/val matrices generated")

			modelseed = random.randint(a=1, b=1000)
			model = xg.XGBRegressor(n_estimators=900,
									learning_rate=0.008,
									max_depth=8,
									subsample=0.18236,
									max_leaves=0,
									seed=modelseed, )


			model.fit(X_train, y_train)
			print("Training complete")

			predictions = model.predict(X_test)
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
						dummy_test_E.append(row[4])  # Energy values are in 5th row
						dummy_predictions.append(predictions[i])

				XS_plotmatrix.append(dummy_test_XS)
				E_plotmatrix.append(dummy_test_E)
				P_plotmatrix.append(dummy_predictions)

			for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
				nuc = validation_nuclides[i]
				current_nuclide = nuc

				evaluation_r2s = []

				truncated_library_r2 = []

				all_library_evaluations = []
				all_predictions = []



				try:
					pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(raw_predictions=pred_xs,
																				  library_xs=true_xs)
				except:
					print(current_nuclide)
				for libxs, p in zip(truncated_endfb, pred_endfb_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)

					every_true_value_list.append(libxs)
					every_prediction_list.append(p)
				evaluation_r2s.append(endfb_r2)

				if current_nuclide in CENDL_nuclides:

					cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL)
					pred_cendl = model.predict(cendl_test)

					pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
					pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl,
																					   library_xs=cendl_xs)
					for libxs, p in zip(truncated_cendl, pred_cendl_gated):
						all_library_evaluations.append(libxs)
						all_predictions.append(p)

						every_true_value_list.append(libxs)
						every_prediction_list.append(p)
					evaluation_r2s.append(pred_cendl_r2)
					truncated_library_r2.append(pred_cendl_r2)
				# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

				if current_nuclide in JENDL_nuclides:
					jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL)
					pred_jendl = model.predict(jendl_test)

					pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
					pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl,
																					   library_xs=jendl_xs)
					for libxs, p in zip(truncated_jendl, pred_jendl_gated):
						all_library_evaluations.append(libxs)
						all_predictions.append(p)

						every_true_value_list.append(libxs)
						every_prediction_list.append(p)
					evaluation_r2s.append(pred_jendl_r2)
					truncated_library_r2.append(pred_jendl_r2)
				# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

				if current_nuclide in JEFF_nuclides:
					jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF)

					pred_jeff = model.predict(jeff_test)

					pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
					pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff,
																					library_xs=jeff_xs)
					for libxs, p in zip(truncated_jeff, pred_jeff_gated):
						all_library_evaluations.append(libxs)
						all_predictions.append(p)

						every_true_value_list.append(libxs)
						every_prediction_list.append(p)

					evaluation_r2s.append(pred_jeff_r2)
					truncated_library_r2.append(pred_jeff_r2)
				# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

				if current_nuclide in TENDL_nuclides:
					tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL)
					pred_tendl = model.predict(tendl_test)

					pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
					pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl,
																					   library_xs=tendl_xs)
					for libxs, p in zip(truncated_tendl, pred_tendl_gated):
						all_library_evaluations.append(libxs)
						all_predictions.append(p)

						every_true_value_list.append(libxs)
						every_prediction_list.append(p)

					evaluation_r2s.append(pred_tendl_r2)
					truncated_library_r2.append(pred_tendl_r2)

				r2 = r2_score(all_library_evaluations, all_predictions)  # various comparisons

				nuclide_r2.append([nuc[0], nuc[1], r2])




			time.sleep(1)
		time_taken = time.time() - time1
		LA += 50
		UA += 50 #
		print(f'completed in {time_taken:0.1f} s.\n')
		print(LA, UA)




	all_libraries_r2 = r2_score(y_true=every_true_value_list, y_pred=every_prediction_list)

	benchmark_r2 = r2_score(every_true_value_list, every_prediction_list)
	run_r2.append(benchmark_r2)
	print(f"R2: {all_libraries_r2:0.5f}")


print('Complete')
print(np.mean(run_r2), np.std(run_r2))
A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]

log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]

r2plot = [i[-1] for i in nuclide_r2]


plt.figure()
plt.plot(A_plots, log_plots, 'x')
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
# plt.hist2d(x=A_plots, y=log_plots, bins=20)
# plt.xlabel("A")
# plt.ylabel("$|\ln(|r^2|)|$")
# plt.colorbar()
# plt.grid()
# plt.show()
#
#
