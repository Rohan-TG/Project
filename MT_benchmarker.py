import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable
import scipy
from matrix_functions import General_plotter, range_setter, mtmake_train, mtmake_test, dsigma_dE, r2_standardiser
import random
# import shap
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error
import time
import datetime
import shap

TENDL = pd.read_csv("TENDL_2021_MT16_with_ENDFB_MT103_107.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=210)

JEFF = pd.read_csv('JEFF33_all_features_MT16_103_107.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=210)

JENDL = pd.read_csv('JENDL5_with_MT16_and_ENDFB_MT103_107.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=210)

CENDL = pd.read_csv('CENDL32_MT16_with_ENDFB_MT103_107.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=210)

scriptrun1 = time.time()

df = pd.read_csv('ENDFBVIII_MT16_91_103_107.csv')
df_test = pd.read_csv('ENDFBVIII_MT16_91_103_107.csv')

df = df[df.Z != 6]
df_test = df_test[df_test.Z != 6]


df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
# df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=0, ua=100, df=df)







validation_set_size = 10
num_runs = 10
run_r2 = []
tally_95 = []
for i in range(num_runs+1):
	nuclides_used = []
	every_prediction_list = []
	every_true_value_list = []
	outlier_tally = 0
	while len(nuclides_used) < len(al):
		time1 = time.time()

		validation_nuclides = []

		while len(validation_nuclides) < validation_set_size:
			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
			if len(nuclides_used) == len(al):
				break

		print(f"{len(nuclides_used)}/{len(al)} selections...")

		X_train, y_train = mtmake_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=100, exclusions=[])

		X_test, y_test = mtmake_test(df= df_test, nuclides=validation_nuclides)

		model = xg.XGBRegressor(n_estimators=900,  # define regressor
								learning_rate=0.008,
								max_depth=8,
								subsample=0.25,
								max_leaves=0,
								seed=42,
								reg_lambda=6,
								)

		model.fit(X_train, y_train)
		print("Training complete...")
		predictions = model.predict(X_test)  # XS predictions
		predictions_ReLU = []

		for pred in predictions:
			if pred >= 0.003:
				predictions_ReLU.append(pred)
			else:
				predictions_ReLU.append(0)

		predictions = predictions_ReLU

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
				pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(raw_predictions=pred_xs, library_xs=true_xs)
			except:
				print(current_nuclide)
			for libxs, p in zip(truncated_endfb, pred_endfb_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				every_prediction_list.append(p)
				every_true_value_list.append(libxs)
			evaluation_r2s.append(endfb_r2)

			if current_nuclide in CENDL_nuclides:

				cendl_test, cendl_xs = mtmake_test(nuclides=[current_nuclide], df=CENDL)
				pred_cendl = model.predict(cendl_test)

				pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl,
																				   library_xs=cendl_xs)
				for libxs, p in zip(truncated_cendl, pred_cendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_prediction_list.append(p)
					every_true_value_list.append(libxs)
				evaluation_r2s.append(pred_cendl_r2)
				truncated_library_r2.append(pred_cendl_r2)
			# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

			if current_nuclide in JENDL_nuclides:
				jendl_test, jendl_xs = mtmake_test(nuclides=[current_nuclide], df=JENDL)
				pred_jendl = model.predict(jendl_test)

				pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
				pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl,
																				   library_xs=jendl_xs)
				for libxs, p in zip(truncated_jendl, pred_jendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_prediction_list.append(p)
					every_true_value_list.append(libxs)
				evaluation_r2s.append(pred_jendl_r2)
				truncated_library_r2.append(pred_jendl_r2)
			# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

			if current_nuclide in JEFF_nuclides:
				jeff_test, jeff_xs = mtmake_test(nuclides=[current_nuclide], df=JEFF)

				pred_jeff = model.predict(jeff_test)

				pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
				pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff,
																				library_xs=jeff_xs)
				for libxs, p in zip(truncated_jeff, pred_jeff_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_prediction_list.append(p)
					every_true_value_list.append(libxs)

				evaluation_r2s.append(pred_jeff_r2)
				truncated_library_r2.append(pred_jeff_r2)
			# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

			if current_nuclide in TENDL_nuclides:
				tendl_test, tendl_xs = mtmake_test(nuclides=[current_nuclide], df=TENDL)
				pred_tendl = model.predict(tendl_test)

				pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
				pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl,
																				   library_xs=tendl_xs)
				for libxs, p in zip(truncated_tendl, pred_tendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)

					limited_evaluations.append(libxs)
					limited_predictions.append(p)

					every_prediction_list.append(p)
					every_true_value_list.append(libxs)

				evaluation_r2s.append(pred_tendl_r2)
				truncated_library_r2.append(pred_tendl_r2)
			# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}"

			# r2 = r2_score(all_library_evaluations, all_predictions)  # various comparisons

			limited_r2 = r2_score(limited_evaluations, limited_predictions)


		for pred in predictions:
			every_prediction_list.append(pred)

		for val in y_test:
			every_true_value_list.append(val)

		for z in evaluation_r2s:
			if z > 0.95:
				outlier_tally += 1
				break

		time_taken = time.time() - time1
		print(f'completed in {time_taken:0.1f} s.\n')

	benchmark_r2 = r2_score(every_true_value_list, every_prediction_list)
	run_r2.append(benchmark_r2)
	print(f"R2: {benchmark_r2:0.5f}")

	print(f"At least one library >= 0.95: {outlier_tally}/{len(al)}")
	tally_95.append(outlier_tally)

	print(f"Total time elapsed: {datetime.timedelta(seconds=(time.time() - scriptrun1))}")

print(run_r2)
print(np.mean(run_r2))
print(np.std(run_r2))