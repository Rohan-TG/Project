import pandas as pd
import numpy as np
import random
import periodictable
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop
from matrix_functions import make_train, make_test, range_setter, r2_standardiser
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("ENDFBVIII_JANIS_features_MT16.csv")

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
	   [18,38], [44,99], [50,126]]

def optimiser(space):

	validation_nuclides = []
	validation_set_size = 20


	for i in range(benchmark_number):

		nuclides_used = []

		while len(validation_nuclides) < validation_set_size:
			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
		print("Nuclide selection complete")

		X_train, y_train = make_train(df=df, exclusions=exc, validation_nuclides=nuclides_used)

		X_test, y_test = make_test(nuclides=validation_nuclides, df=df)
		print("Data prep done")

		model = xg.XGBRegressor(**space, seed=42)

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