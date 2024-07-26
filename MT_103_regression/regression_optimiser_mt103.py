import random

import numpy as np
import xgboost as xg
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error
import hyperopt.early_stop
from matrix_functions import range_setter
import MT_103_functions

df = pd.read_csv('ENDFBVIII_MT103_fund_features_only.csv')
al = range_setter(df, la= 30, ua=210)

jeff = pd.read_csv('JEFF-3.3_MT103_fund_only.csv')
jeffnuclides= range_setter(df=jeff, la=30, ua=208)

jendl = pd.read_csv('JENDL-5_MT103_fund_features_only.csv')
jendlnuclides = range_setter(df=jendl, la=30, ua=208)

tendl = pd.read_csv('TENDL-2021_MT103_fund_only.csv')
tendlnuclides = range_setter(df=tendl, la=30, ua=208)

cendl = pd.read_csv('CENDL-3.2_MT103_fund_features_only.csv')
cendlnuclides = range_setter(df=cendl, la=30, ua=208)


estimators = np.arange(50, 2000, 50)
space = {'n_estimators': hp.choice('n_estimators', estimators),
		 'subsample': hp.uniform('subsample', 0.01, 0.99),
		 'max_leaves': 0,
		 'max_depth': scope.int(hp.quniform('max_depth', 5,12,1)),
		 'learning_rate': hp.uniform('learning_rate', 0.0005, 0.5)}









def optimiser(s):

	validation_set_size = 50
	validation_nuclides = []

	while len(validation_nuclides) <validation_set_size:
		choice = random.choice(al)
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)


	X_train, y_train = MT_103_functions.make_train(df=df, la=30, ua=208, validation_nuclides=validation_nuclides)

	X_test, y_test = MT_103_functions.make_test(df=df, nuclides=validation_nuclides)

	model = xg.XGBRegressor(**s)

	evaluation = [(X_train, y_train), (X_test, y_test)]

	model.fit(X_train, y_train,
			  eval_set=evaluation,
			  eval_metric='rmse',
			  verbose=False)

	predictions = model.predict(X_test)

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
				dummy_test_E.append(row[3])  # Energy values are in 5th row
				dummy_predictions.append(predictions[i])

		XS_plotmatrix.append(dummy_test_XS)
		E_plotmatrix.append(dummy_test_E)
		P_plotmatrix.append(dummy_predictions)

	try:

		for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
			all_preds = []
			all_libs = []
			current_nuclide = validation_nuclides[i]

			nuc = validation_nuclides[i]  # validation nuclide

			if current_nuclide in cendlnuclides:

				cendl_test, cendl_xs = MT_103_functions.make_test(nuclides=[current_nuclide], df=cendl)
				pred_cendl = model.predict(cendl_test)

				pred_cendl_gated, truncated_cendl, pred_cendl_rmse = MT_103_functions.rmse_standard(libxs=cendl_xs, pxs=pred_cendl)

				for x, y in zip(pred_cendl_gated, truncated_cendl):
					all_libs.append(y)
					all_preds.append(x)

			if current_nuclide in jendlnuclides:

				jendl_test, jendl_xs = MT_103_functions.make_test(nuclides=[current_nuclide], df=jendl)
				pred_jendl = model.predict(jendl_test)

				pred_jendl_gated, truncated_jendl, pred_jendl_rmse = MT_103_functions.rmse_standard(pxs=pred_jendl,
																				   libxs=jendl_xs)

				for x, y in zip(pred_jendl_gated, truncated_jendl):
					all_libs.append(y)
					all_preds.append(x)

			if current_nuclide in jeffnuclides:
				jeff_test, jeff_xs = MT_103_functions.make_test(nuclides=[current_nuclide], df=jeff)

				pred_jeff = model.predict(jeff_test)

				pred_jeff_gated, truncated_jeff, pred_jeff_rmse = MT_103_functions.rmse_standard(pxs=pred_jeff,
																				libxs=jeff_xs)
				for x, y in zip(pred_jeff_gated, truncated_jeff):
					all_libs.append(y)
					all_preds.append(x)

			if current_nuclide in tendlnuclides:

				tendl_test, tendl_xs = MT_103_functions.make_test(nuclides=[current_nuclide], df=tendl)
				pred_tendl = model.predict(tendl_test)

				pred_tendl_gated, truncated_tendl, pred_tendl_rmse = MT_103_functions.rmse_standard(pxs=pred_tendl,
																				   libxs=tendl_xs)
				for x, y in zip(pred_tendl_gated, truncated_tendl):
					all_libs.append(y)
					all_preds.append(x)

	except:
		print(current_nuclide)
		return(1)

	HP_SET_RMSE = mean_squared_error(all_libs, all_preds) ** 0.5

	return({'loss': HP_SET_RMSE, 'status': STATUS_OK, 'model': model})


trials = Trials()

best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=100,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(50))

bestmodel = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(bestmodel)