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
import shap
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop

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



df = pd.read_csv('ENDFBVIII_MT16_MT103_MT107_fund_features.csv')
df_test = pd.read_csv('ENDFBVIII_MT16_MT103_MT107_fund_features.csv')
df = df[df.Z != 6]
df_test = df_test[df_test.Z != 6]

df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
# df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=0, ua=100, df=df)


space = {'n_estimators': hp.choice('n_estimators', [100, 200, 250, 275, 300,
													325, 350, 400, 450, 500, 600, 650, 700, 750, 800, 850, 900, 1000, 1100, 1200, ]),
		 'subsample': hp.loguniform('subsample', np.log(0.05), np.log(1.0)),
		 'max_leaves': 0,
		 'max_depth': scope.int(hp.quniform("max_depth", 4, 12, 1)),
		 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
		 'reg_lambda': hp.choice('lambda', [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3])}


def optimiser(space):

	validation_nuclides = []
	validation_set_size = 15

	# random.seed(a=10)

	while len(validation_nuclides) < validation_set_size: # up to 25 nuclides
		choice = random.choice(al) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("Test nuclide selection complete")







	X_train, y_train = mtmake_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=100,) # create training matrix
	X_test, y_test = mtmake_test(validation_nuclides, df=df_test,) # create test matrix using validation nuclides
	print("Data prep done")

	model = xg.XGBRegressor(**space, seed=42)

	evaluation = [(X_train, y_train), (X_test, y_test)]

	model.fit(X_train, y_train,
			  eval_set=evaluation, eval_metric='rmse')

	print("Training complete")
	predictions = model.predict(X_test)  # XS predictions
	predictions_ReLU = []

	for pred in predictions: # prediction gate
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
				dummy_test_E.append(row[4])  # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	all_preds = []
	all_libs = []
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):

		current_nuclide = validation_nuclides[i]

		# jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
		# cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
		# jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
		# tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])


		nuc = validation_nuclides[i]  # validation nuclide
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

		# mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)
		# print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
		# time.sleep(0.8)

		pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
		pred_endfb_gated, truncated_endfb, pred_endfb_r2 = r2_standardiser(raw_predictions=pred_xs, library_xs=true_xs)
		for x, y in zip(pred_endfb_gated, truncated_endfb):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")
	print("Run complete \n")


	HP_SET_MSE = mean_squared_error(all_preds, all_libs) # for one iteration of the k-fold cross validation

	return {'loss': HP_SET_MSE, 'status': STATUS_OK, 'model': model}

trials = Trials()

best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=1000,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(100))

# print(best)

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)
