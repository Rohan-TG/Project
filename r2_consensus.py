import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable
import scipy
from matrix_functions import General_plotter, range_setter, make_train, make_test, dsigma_dE
import random
import shap
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error
import time
import tqdm
TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=260)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=260)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=260)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=260)


df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")


df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
# df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=30, ua=210, df=df)





n_evaluations = 100
all_r2_list = []
target_nuclide = [26,56]

for i in tqdm.tqdm(range(n_evaluations)):
	print(f"\nRun {i + 1}/{n_evaluations}")


	validation_nuclides = [target_nuclide]
	validation_set_size = 10

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al) # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("Test nuclide selection complete")



	X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=210,)
	X_test, y_test = make_test([target_nuclide], df=df_test,)
	print("Data prep done")

	model_seed = random.randint(a=1, b=1000)

	model = xg.XGBRegressor(n_estimators=500,
							learning_rate=0.01,
							max_depth=8,
							subsample=0.2,
							max_leaves=0,
							seed=model_seed, )


	model.fit(X_train, y_train)
	print("Training complete")


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


	dummy_test_XS = []
	dummy_test_E = []
	dummy_predictions = []
	for i, row in enumerate(X_test):
		if [row[0], row[1]] == target_nuclide:
			dummy_test_XS.append(y_test[i])
			dummy_test_E.append(row[4])  # Energy values are in 5th row
			dummy_predictions.append(predictions[i])

	XS_plotmatrix.append(dummy_test_XS)
	E_plotmatrix.append(dummy_test_E)
	P_plotmatrix.append(dummy_predictions)

	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		all_preds = []
		all_libs = []
		# current_nuclide = validation_nuclides[i]


		# nuc = validation_nuclides[i]  # validation nuclide
		jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[target_nuclide])
		cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[target_nuclide])
		jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[target_nuclide])
		tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[target_nuclide])

		# nuc = validation_nuclides[i]  # validation nuclide
		# plt.plot(erg, pred_xs, label='Predictions', color='red')
		# plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
		# plt.plot(tendlerg, tendlxs, label="TENDL21", color='dimgrey', linewidth=2)
		# if nuc in JEFF_nuclides:
		# 	plt.plot(jefferg, jeffxs, '--', label='JEFF3.3', color='mediumvioletred')
		# if nuc in JENDL_nuclides:
		# 	plt.plot(jendlerg, jendlxs, label='JENDL5', color='green')
		# if nuc in CENDL_nuclides:
		# 	plt.plot(cendlerg, cendlxs, '--', label='CENDL3.2', color='gold')
		# plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
		# plt.legend()
		# plt.grid()
		# plt.ylabel('$\sigma_{n,2n}$ / b')
		# plt.xlabel('Energy / MeV')
		# plt.show()

		pred_endfb_r2 = r2_score(y_true=true_xs, y_pred=pred_xs)
		for x, y in zip(pred_xs, true_xs):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} ")

		if target_nuclide in CENDL_nuclides:

			cendl_test, cendl_xs = make_test(nuclides=[target_nuclide], df=CENDL)
			pred_cendl = model.predict(cendl_test)

			pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=pred_cendl)
			for x, y in zip(pred_cendl, cendl_xs):
				all_libs.append(y)
				all_preds.append(x)
			print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f}")

		if target_nuclide in JENDL_nuclides:

			jendl_test, jendl_xs = make_test(nuclides=[target_nuclide], df=JENDL)
			pred_jendl = model.predict(jendl_test)

			pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=pred_jendl)
			print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f}")
			for x, y in zip(pred_jendl, jendl_xs):
				all_libs.append(y)
				all_preds.append(x)

		if target_nuclide in JEFF_nuclides:
			jeff_test, jeff_xs = make_test(nuclides=[target_nuclide], df=JEFF)

			pred_jeff = model.predict(jeff_test)

			pred_jeff_r2 = r2_score(y_true=jeff_xs, y_pred=pred_jeff)
			for x, y in zip(pred_jeff, jeff_xs):
				all_libs.append(y)
				all_preds.append(x)
			print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f}")

		if target_nuclide in TENDL_nuclides:

			tendl_test, tendl_xs = make_test(nuclides=[target_nuclide], df=TENDL)
			pred_tendl = model.predict(tendl_test)

			pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
			pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=pred_tendl)
			for x, y in zip(pred_tendl, tendl_xs):
				all_libs.append(y)
				all_preds.append(x)
			print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f}")
		consensus = r2_score(all_libs, all_preds)
		print(f"Consensus R2: {r2_score(all_libs, all_preds):0.5f}")
		all_r2_list.append(consensus)

print(all_r2_list)

mu = np.mean(all_r2_list)
sigma = np.std(all_r2_list)
interval_low, interval_high = scipy.stats.t.interval(0.95, loc=mu, scale=sigma, df=(len(all_r2_list) - 1))
print(f"\nConsensus: {mu:0.5f} +/- {interval_high-interval_low:0.5f}")