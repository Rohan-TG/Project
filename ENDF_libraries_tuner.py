import pandas as pd
import numpy as np
import random
import scipy
import periodictable
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop
from matrix_functions import range_setter, General_plotter, make_test, make_train
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error

ENDFBVIII = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
TENDL = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
CENDL = pd.read_csv('CENDL32_features_arange_zeroed.csv')

CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)
nucs = range_setter(df=ENDFBVIII, la=30, ua=210)



space = {'n_estimators': hp.choice('n_estimators', [500, 600, 700, 800, 850, 900, 950, 1000, 1100,]),
		 'subsample': hp.loguniform('subsample', np.log(0.05), np.log(1.0)),
		 'max_leaves': 0,
		 'max_depth': scope.int(hp.quniform("max_depth", 6, 12, 1)),
		 # 'reg_lambda': hp.loguniform('lambda', np.log(0.5), np.log(3.0)),
		 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.25)),
		 'gamma': hp.loguniform('gamma', np.log(0.1), np.log(5))}

def optimiser(space):

	validation_nuclides = []
	validation_set_size = 50 # number of nuclides hidden from training

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(nucs) # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides:
			validation_nuclides.append(choice)
	print("Test nuclide selection complete")

	X_train, y_train = make_train(df=ENDFBVIII, validation_nuclides=validation_nuclides, la=30, ua=215)
	X_test, y_test = make_test(validation_nuclides, df=ENDFBVIII)

	print("Data prep done")

	model = xg.XGBRegressor(**space, seed=42)

	evaluation = [(X_train, y_train), (X_test, y_test)]

	model.fit(X_train, y_train, eval_set= evaluation, eval_metric='rmse')

	print("Training complete")

	predictions = model.predict(X_test)  # XS predictions
	print("Processing evaluations...")

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

	batch_r2 = []
	batch_mse = []
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i]

		evaluation_r2s = []
		evaluation_mse = []
		evaluation_r2s.append(r2_score(y_true=true_xs, y_pred=pred_xs))

		pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
		pred_endfb_r2 = r2_score(y_true=true_xs, y_pred=pred_xs)
		evaluation_r2s.append(pred_endfb_r2)
		evaluation_mse.append(pred_endfb_mse)
		# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f}")

		f_pred = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')

		if nuc in CENDL_nuclides:
			cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[nuc])

			predictions_interpolated_cendl = f_pred(cendl_erg)

			pred_cendl_mse = mean_squared_error(predictions_interpolated_cendl, cendl_xs)
			pred_cendl_r2 = r2_score(y_true=cendl_xs, y_pred=predictions_interpolated_cendl)
			evaluation_r2s.append(pred_cendl_r2)
			evaluation_mse.append(pred_cendl_mse)

		if nuc in JENDL_nuclides:
			jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[nuc])

			predictions_interpolated_jendl = f_pred(jendl_erg)

			pred_jendl_mse = mean_squared_error(predictions_interpolated_jendl, jendl_xs)
			pred_jendl_r2 = r2_score(y_true=jendl_xs, y_pred=predictions_interpolated_jendl)
			evaluation_r2s.append(pred_jendl_r2)
			evaluation_mse.append(pred_jendl_mse)
		# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

		if nuc in JEFF_nuclides:
			jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[nuc])

			predictions_interpolated_jeff = f_pred(jeff_erg)

			pred_jeff_mse = mean_squared_error(predictions_interpolated_jeff, jeff_xs)
			pred_jeff_r2 = r2_score(y_true=jeff_xs, y_pred=predictions_interpolated_jeff)
			evaluation_r2s.append(pred_jeff_r2)
			evaluation_mse.append(pred_jeff_mse)

		if nuc in TENDL_nuclides:
			tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[nuc])
			predictions_interpolated_tendl = f_pred(tendl_erg)

			pred_tendl_mse = mean_squared_error(predictions_interpolated_tendl, tendl_xs)
			pred_tendl_r2 = r2_score(y_true=tendl_xs, y_pred=predictions_interpolated_tendl)
			evaluation_r2s.append(pred_tendl_r2)
			evaluation_mse.append(pred_tendl_mse)
		# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

		mean_nuclide_r2 = np.mean(evaluation_r2s)  # r2 for nuclide nuc
		mean_nuclide_mse = np.mean(evaluation_mse)

		batch_mse.append(mean_nuclide_mse)

		batch_r2.append(mean_nuclide_r2)

		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]} overall R2: {mean_nuclide_r2:0.6f}")

	r2 = np.mean(batch_r2)

	mse_overall = np.mean(batch_mse)

	r2_return = 1 - r2

	return {'loss': mse_overall, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=optimiser,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=250,
            early_stop_fn=hyperopt.early_stop.no_progress_loss(100))

print(best)


best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
print(best_model)