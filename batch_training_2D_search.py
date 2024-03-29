import pandas as pd
import matplotlib.pyplot as plt
import periodictable
import scipy
from matrix_functions import General_plotter, range_setter, make_train, make_test, dsigma_dE, r2_standardiser
import random
import shap
import numpy as np
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import cm
import time
import tqdm


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
df.index = range(len(df))



tempal = range_setter(la=0, ua=100, df=df)


banned_nuclides = [[3, 7],
[4, 9],
[1, 2],
[1, 3],
[7, 14]]

al = []
for i in tempal:
	if i not in banned_nuclides:
		al.append(i)

validation_set_size = 1


search_space = []
search_square = 5

for p in range(1, search_square + 1):
	for q in range(1, search_square + 1):
		search_space.append([p*5, q*5])

record_list = []
for space in tqdm.tqdm(search_space):
	nuclides_used = []
	nuclide_r2 = []


	lower_bound = space[0]
	upper_bound = space[1]
	benchmark_libxs = []
	benchmark_predictions = []


	while len(nuclides_used) < len(al):
		time1 = time.time()

		validation_nuclide = random.choice(al)
		validation_nuclides = [validation_nuclide]
		nuclides_used.append(validation_nuclide)
		LA = validation_nuclide[1] - lower_bound
		UA = validation_nuclide[1] + upper_bound


		print(f"{len(nuclides_used)}/{len(al)} selections")

		X_train, y_train = make_train(df=df, validation_nuclides=[validation_nuclide], ua=UA, la=LA)

		X_test, y_test = make_test(nuclides=[validation_nuclide], df=df)

		# print("Train/val matrices generated")

		modelseed = random.randint(a=1, b=1000)
		model = xg.XGBRegressor(n_estimators=900,
								learning_rate=0.008,
								max_depth=8,
								subsample=0.18236,
								max_leaves=0,
								seed=modelseed, )


		model.fit(X_train, y_train)
		# print("Training complete")

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


			limited_evaluations = []
			limited_predictions = []

			every_prediction_list_single_nuclide = []
			every_true_value_list_single_nuclide = []

			try:
				pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(raw_predictions=pred_xs,
																			  library_xs=true_xs)
			except:
				print(current_nuclide)
			for libxs, p in zip(truncated_endfb, pred_endfb_gated):
				benchmark_libxs.append(libxs)
				benchmark_predictions.append(p)

				every_true_value_list_single_nuclide.append(libxs)
				every_prediction_list_single_nuclide.append(p)
			evaluation_r2s.append(endfb_r2)

			if current_nuclide in CENDL_nuclides:

				cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL)
				pred_cendl = model.predict(cendl_test)

				pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl,
																				   library_xs=cendl_xs)
				for libxs, p in zip(truncated_cendl, pred_cendl_gated):
					benchmark_libxs.append(libxs)
					benchmark_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_true_value_list_single_nuclide.append(libxs)
					every_prediction_list_single_nuclide.append(p)
				evaluation_r2s.append(pred_cendl_r2)
				truncated_library_r2.append(pred_cendl_r2)
			# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

			if current_nuclide in JENDL_nuclides:
				jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL)
				pred_jendl = model.predict(jendl_test)

				pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl,
																				   library_xs=jendl_xs)
				for libxs, p in zip(truncated_jendl, pred_jendl_gated):
					benchmark_libxs.append(libxs)
					benchmark_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_true_value_list_single_nuclide.append(libxs)
					every_prediction_list_single_nuclide.append(p)
				evaluation_r2s.append(pred_jendl_r2)
				truncated_library_r2.append(pred_jendl_r2)
			# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

			if current_nuclide in JEFF_nuclides:
				jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF)

				pred_jeff = model.predict(jeff_test)

				pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff,
																				library_xs=jeff_xs)
				for libxs, p in zip(truncated_jeff, pred_jeff_gated):
					benchmark_libxs.append(libxs)
					benchmark_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_true_value_list_single_nuclide.append(libxs)
					every_prediction_list_single_nuclide.append(p)

				evaluation_r2s.append(pred_jeff_r2)
				truncated_library_r2.append(pred_jeff_r2)
			# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

			if current_nuclide in TENDL_nuclides:
				tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL)
				pred_tendl = model.predict(tendl_test)

				pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl,
																				   library_xs=tendl_xs)
				for libxs, p in zip(truncated_tendl, pred_tendl_gated):
					benchmark_libxs.append(libxs)
					benchmark_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_true_value_list_single_nuclide.append(libxs)
					every_prediction_list_single_nuclide.append(p)

				evaluation_r2s.append(pred_tendl_r2)
				truncated_library_r2.append(pred_tendl_r2)

			r2 = r2_score(every_true_value_list_single_nuclide,every_prediction_list_single_nuclide)  # various comparisons

			# limited_r2 = r2_score(limited_evaluations, limited_predictions)

			nuclide_r2.append([nuc[0], nuc[1], r2])


		time_taken = time.time() - time1
		print(f'completed in {time_taken:0.1f} s.\n')


	all_libraries_r2 = r2_score(y_true=benchmark_libxs, y_pred= benchmark_predictions)

	record_list.append([space, all_libraries_r2])
	print(f"Search params: {space, all_libraries_r2}")
	print(f"R2: {all_libraries_r2:0.5f} \n")



A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]

log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]

# r2plot = [i[-1] for i in nuclide_r2]



lower_plots = [i+1 for i in range(search_square)]
upper_plots = [i+1 for i in range(search_square)]
lower_plots, upper_plots = np.meshgrid(lower_plots, upper_plots)


r2plots = [i[1] for i in record_list]
r2plots = np.array(r2plots)
r2plots_matrix = r2plots.reshape(search_square, search_square)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(lower_plots, upper_plots, r2plots_matrix, cmap= cm.coolwarm,
				 linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# plt.figure()
# plt.plot(A_plots, log_plots, 'x')
# plt.xlabel("A")
# plt.ylabel("$|\ln(|r^2|)|$")
# plt.title("Performance - A")
# plt.grid()
# plt.show()
#
# plt.figure()
# plt.plot(Z_plots, log_plots_Z, 'x')
# plt.xlabel('Z')
# plt.ylabel("$|\ln(|r^2|)|$")
# plt.title("Performance - Z")
# plt.grid()
# plt.show()
#
#
#
# plt.figure()
# plt.hist2d(x=A_plots, y=log_plots, bins=20)
# plt.xlabel("A")
# plt.ylabel("$|\ln(|r^2|)|$")
# plt.colorbar()
# plt.grid()
# plt.show()


