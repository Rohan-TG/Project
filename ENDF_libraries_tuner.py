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
	for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
		nuc = validation_nuclides[i]

		evaluation_r2s = []
		evaluation_r2s.append(r2_score(y_true=true_xs, y_pred=pred_xs))
		if nuc in nucs:
			endfb8_erg, endfb8_xs = General_plotter(df=ENDFBVIII, nuclides=[nuc])

			x_interpolate_endfb8 = np.linspace(start=0.00000000, stop=max(endfb8_erg), num=500)
			f_pred_endfb8 = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
			predictions_interpolated_endfb8 = f_pred_endfb8(x_interpolate_endfb8)

			f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg, y=endfb8_xs, fill_value='extrapolate')
			endfb8_interp_xs = f_endfb8(x_interpolate_endfb8)
			pred_endfb_r2 = r2_score(y_true=endfb8_interp_xs, y_pred=predictions_interpolated_endfb8)
			evaluation_r2s.append(pred_endfb_r2)
		# print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f}")

		if nuc in CENDL_nuclides:
			cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[nuc])

			x_interpolate_cendl = np.linspace(start=0.000000001, stop=max(cendl_erg), num=500)
			f_pred_cendl = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
			predictions_interpolated_cendl = f_pred_cendl(x_interpolate_cendl)

			f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs)
			cendl_interp_xs = f_cendl32(x_interpolate_cendl)
			pred_cendl_r2 = r2_score(y_true=cendl_interp_xs, y_pred=predictions_interpolated_cendl)
			evaluation_r2s.append(pred_cendl_r2)

		if nuc in JENDL_nuclides:
			jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[nuc])

			x_interpolate_jendl = np.linspace(start=0.000000001, stop=max(jendl_erg), num=500)
			f_pred_jendl = scipy.interpolate.interp1d(x=erg, y=pred_xs)
			predictions_interpolated_jendl = f_pred_jendl(x_interpolate_jendl)

			f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs)
			jendl_interp_xs = f_jendl5(x_interpolate_jendl)
			pred_jendl_r2 = r2_score(y_true=jendl_interp_xs, y_pred=predictions_interpolated_jendl)
			evaluation_r2s.append(pred_jendl_r2)
		# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

		if nuc in JEFF_nuclides:
			jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[nuc])

			x_interpolate_jeff = np.linspace(start=0.000000001, stop=max(jeff_erg), num=500)
			f_pred_jeff = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
			predictions_interpolated_jeff = f_pred_jeff(x_interpolate_jeff)

			f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs)
			jeff_interp_xs = f_jeff33(x_interpolate_jeff)
			pred_jeff_r2 = r2_score(y_true=jeff_interp_xs, y_pred=predictions_interpolated_jeff)
			evaluation_r2s.append(pred_jeff_r2)

		if nuc in TENDL_nuclides:
			tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[nuc])

			x_interpolate_tendl = np.linspace(start=0.000000001, stop=max(tendl_erg), num=500)
			f_pred_tendl = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
			predictions_interpolated_tendl = f_pred_tendl(x_interpolate_tendl)

			f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs)
			tendl_interp_xs = f_tendl21(x_interpolate_tendl)
			pred_tendl_r2 = r2_score(y_true=tendl_interp_xs, y_pred=predictions_interpolated_tendl)
			evaluation_r2s.append(pred_tendl_r2)
		# print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

		mean_nuclide_r2 = np.mean(evaluation_r2s)  # r2 for nuclide nuc

		batch_r2.append(mean_nuclide_r2)

		print(f"{periodictable.elements[nuc[0]]}-{nuc[1]} overall R2: {mean_nuclide_r2:0.6f}")

	r2 = np.mean(batch_r2)

	r2_return = 1 - r2

	return {'loss': r2_return, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=optimiser,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=150,
            early_stop_fn=hyperopt.early_stop.no_progress_loss(60))

print(best)


best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
print(best_model)