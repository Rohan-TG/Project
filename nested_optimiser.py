import pandas as pd
import numpy as np
import random
import periodictable
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop
from matrix_functions import make_train, make_test, range_setter, r2_standardiser, exclusion_func
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("ENDFBVIII_JANIS_features_MT16.csv")

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)

TENDL = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL_nuclides = range_setter(df=CENDL,la=30, ua=210)


al = range_setter(df=df, la=30, ua=210)


space = {'n_estimators': hp.choice('n_estimators', [500, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1200, 1250, 1300, 1350, 1400]),
		 'subsample': hp.loguniform('subsample', np.log(0.05), np.log(1.0)),
		 'max_leaves': 0,
		 'max_depth': scope.int(hp.quniform("max_depth", 6, 12, 1)),
		 'learning_rate': hp.loguniform('learning_rate', np.log(0.0005), np.log(0.03))}


benchmark_number = 20
nuclide_mse = []
nuclide_r2 = []

every_prediction_list = []
every_true_value_list = []

exc = exclusion_func()

def optimiser(space):

	validation_nuclides = []
	validation_set_size = 20


	for i in range(benchmark_number):

		every_prediction_list = []
		every_true_value_list = []
		potential_outliers = []

		benchmark_total_library_evaluations = []
		benchmark_total_predictions = []

		nuclides_used = []

		while len(validation_nuclides) < validation_set_size:
			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
		print("Nuclide selection complete")

		X_train, y_train = make_train(df=df, exclusions=exc,
									  la=30, ua=210,
									  validation_nuclides=nuclides_used)

		X_test, y_test = make_test(nuclides=validation_nuclides, df=df)

		model = xg.XGBRegressor(**space, seed=42,
								early_stopping_rounds=10)

		evaluation = [(X_train, y_train), (X_test, y_test)]

		model.fit(X_train, y_train,
				  eval_set=evaluation, eval_metric='rmse',
				  verbose = True)

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
					dummy_test_E.append(row[4]) # Energy values are in 5th row
					dummy_predictions.append(predictions[i])

			XS_plotmatrix.append(dummy_test_XS)
			E_plotmatrix.append(dummy_test_E)
			P_plotmatrix.append(dummy_predictions)

		for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):

			nuc = validation_nuclides[i] # validation nuclide
			current_nuclide = nuc
			evaluation_r2s = []
			truncated_library_r2 = []
			all_library_evaluations = []
			all_predictions = []
			r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
			print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")

			mse = mean_squared_error(true_xs, pred_xs)

			nuclide_mse.append([nuc[0], nuc[1], mse])
			nuclide_r2.append([nuc[0], nuc[1], r2])

			overall_r2 = r2_score(y_test, predictions)
			for pred in predictions:
				every_prediction_list.append(pred)

			for val in y_test:
				every_true_value_list.append(val)

			try:
				pred_endfb_gated, truncated_endfb, endfb_r2 = r2_standardiser(raw_predictions=pred_xs,
																				   library_xs=true_xs)
			except:
				print(current_nuclide)
			for libxs, p in zip(truncated_endfb, pred_endfb_gated):
				all_library_evaluations.append(libxs)
				all_predictions.append(p)

				benchmark_total_library_evaluations.append(libxs)
				benchmark_total_predictions.append(p)

			evaluation_r2s.append(endfb_r2)

			if current_nuclide in CENDL_nuclides:

				cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL)
				pred_cendl = model.predict(cendl_test)

				pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
				pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl, library_xs=cendl_xs)
				for libxs, p in zip(truncated_cendl, pred_cendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)


					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)

				evaluation_r2s.append(pred_cendl_r2)
				truncated_library_r2.append(pred_cendl_r2)
				# print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

			if current_nuclide in JENDL_nuclides:
				jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL)
				pred_jendl = model.predict(jendl_test)

				pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
				pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl, library_xs=jendl_xs)
				for libxs, p in zip(truncated_jendl, pred_jendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)

					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)
				evaluation_r2s.append(pred_jendl_r2)
				truncated_library_r2.append(pred_jendl_r2)
				# print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

			if current_nuclide in JEFF_nuclides:
				jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF)

				pred_jeff = model.predict(jeff_test)

				pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
				pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff, library_xs=jeff_xs)
				for libxs, p in zip(truncated_jeff, pred_jeff_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)


					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)

				evaluation_r2s.append(pred_jeff_r2)
				truncated_library_r2.append(pred_jeff_r2)
				# print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

			if current_nuclide in TENDL_nuclides:
				tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL)
				pred_tendl = model.predict(tendl_test)

				pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
				pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl, library_xs=tendl_xs)
				for libxs, p in zip(truncated_tendl, pred_tendl_gated):
					all_library_evaluations.append(libxs)
					all_predictions.append(p)


					benchmark_total_library_evaluations.append(libxs)
					benchmark_total_predictions.append(p)

				evaluation_r2s.append(pred_tendl_r2)
				truncated_library_r2.append(pred_tendl_r2)

	HP_SET_R2 = r2_standardiser(raw_predictions=every_prediction_list, library_xs=every_true_value_list) # for one iteration of the k-fold cross validation

	target_r2 = 1 - HP_SET_R2

	return {'loss': target_r2, 'status': STATUS_OK, 'model': model}

trials = Trials()

best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=100,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(20))

# print(best)

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)