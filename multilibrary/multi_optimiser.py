from matrix_functions import make_test, make_train, range_setter, General_plotter, r2_standardiser
import pandas as pd
import xgboost as xg
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import time
import shap
import numpy as np
import periodictable

time1 = time.time()

ENDFB = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
TENDL = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JEFF = pd.read_csv('JEFF33_all_features.csv')
CENDL = pd.read_csv('CENDL32_all_features.csv')

ENDFB_nuclides = range_setter(df=ENDFB, la=30, ua=210)
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)
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
	   [18,38], [44,99], [50,126]] # 10 sigma with handpicked additions
all_libraries = pd.read_csv('MT16_all_libraries_mk1.csv')
all_libraries.index = range(len(all_libraries))

space = {'n_estimators': hp.choice('n_estimators', [800, 900, 1000, 1100, 1200, 1300, 1400, 1800, 1900, 2000, 2200]),
		 'subsample': hp.loguniform('subsample', np.log(0.05), np.log(1.0)),
		 'max_leaves': 0,
		 'max_depth': scope.int(hp.quniform("max_depth", 6, 12, 1)),
		 'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.08))}


benchmark_number = 1
nuclide_r2 = []
every_prediction_list = []
every_true_value_list = []



def optimiser(space):

	validation_nuclides = []
	validation_set_size = 20

	nuclides_used = []

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(ENDFB_nuclides)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides and choice not in nuclides_used:
			validation_nuclides.append(choice)
			nuclides_used.append(choice)
	print("Nuclide selection complete. Forming matrices...")

	X_train, y_train = make_train(df=all_libraries, la=30, ua=210, validation_nuclides=validation_nuclides,
								  exclusions=exc)

	X_test, y_test = make_test(nuclides=validation_nuclides, df=ENDFB)
	print("Data preparation complete. Training...")

	model = xg.XGBRegressor(**space, seed=42)

	evaluation = [(X_train, y_train), (X_test, y_test)]

	model.fit(X_train, y_train,
			  eval_set=evaluation, eval_metric='rmse')

	print('Training complete. Evaluating...')

	predictions = model.predict(X_test)
	predictions_ReLU = []

	for pred in predictions:  # prediction gate
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

	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		all_preds = []
		all_libs = []
		current_nuclide = validation_nuclides[i]

		nuc = validation_nuclides[i]  # validation nuclide

		pred_endfb_gated, truncated_endfb, pred_endfb_r2 = r2_standardiser(raw_predictions=pred_xs,
																		   library_xs=true_xs)

		for x, y in zip(pred_endfb_gated, truncated_endfb):
			all_libs.append(y)
			all_preds.append(x)
		# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MAPE: {endfbmape:0.4f}")

		# Find r2 w.r.t. other libraries
		if current_nuclide in CENDL_nuclides:

			cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL)
			pred_cendl = model.predict(cendl_test)

			pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl,
																			   library_xs=cendl_xs)

			for x, y in zip(pred_cendl_gated, truncated_cendl):
				all_libs.append(y)
				all_preds.append(x)
			# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

		if current_nuclide in JENDL_nuclides:

			jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL)
			pred_jendl = model.predict(jendl_test)

			pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl,
																			   library_xs=jendl_xs)

			for x, y in zip(pred_jendl_gated, truncated_jendl):
				all_libs.append(y)
				all_preds.append(x)
			# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

		if current_nuclide in JEFF_nuclides:
			jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF)

			pred_jeff = model.predict(jeff_test)

			pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff,
																			library_xs=jeff_xs)
			for x, y in zip(pred_jeff_gated, truncated_jeff):
				all_libs.append(y)
				all_preds.append(x)
			# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

		if current_nuclide in TENDL_nuclides:

			tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL)
			pred_tendl = model.predict(tendl_test)

			pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl,
																			   library_xs=tendl_xs)
			for x, y in zip(pred_tendl_gated, truncated_tendl):
				all_libs.append(y)
				all_preds.append(x)
			# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

	HP_SET_R2 = r2_score(all_libs, all_preds) # for one iteration of the k-fold cross validation

	target_r2 = 1 - HP_SET_R2

	return {'loss': target_r2, 'status': STATUS_OK, 'model': model}



trials = Trials()

best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=70,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(50))

print(best)
print('==========================================================')

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)