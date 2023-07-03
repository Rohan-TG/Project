import pandas as pd
import numpy as np
import random
import periodictable
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop
from matrix_functions import anomaly_remover, OPT_make_train, OPT_make_test
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("ENDFBVIII_zeroed_LDP_XS.csv")
df_test = pd.read_csv("ENDFBVIII_zeroed_LDP_XS.csv")

al = []
for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
	if [j, i] in al or i > 210:
		continue
	else:
		al.append([j, i]) # format is [Z, A]



space = {'n_estimators': hp.choice('n_estimators', [500, 600, 650, 700, 750, 800, 850, 900, 1000, 1100, 1200, ]),
		 'subsample': hp.loguniform('subsample', np.log(0.05), np.log(1.0)),
		 'max_leaves': 0,
		 'max_depth': scope.int(hp.quniform("max_depth", 6, 12, 1)),
		 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.03))}



benchmark_number = 3
nuclide_mse = []
nuclide_r2 = []

every_prediction_list = []
every_true_value_list = []



def optimiser(space):

	validation_nuclides = []
	validation_set_size = 10

	test_nuclides = [[26, 56], [78, 195], [24, 52], [92, 235], [92, 238],
					 [40, 90], ]


	for i in range(benchmark_number):

		nuclides_used = []

		while len(validation_nuclides) < validation_set_size:
			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
		print("Nuclide selection complete")

		X_train, y_train = OPT_make_train(df=df, val_nuc=validation_nuclides, test_nuc=test_nuclides)

		X_test, y_test = OPT_make_test(nuclides=validation_nuclides, df=df_test)
		print("Data prep done")

		model = xg.XGBRegressor(**space, seed=42)

		evaluation = [(X_train, y_train), (X_test, y_test)]

		model.fit(X_train, y_train,
				  eval_set=evaluation, eval_metric='rmse')

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

	HP_SET_R2 = r2_score(every_true_value_list, every_prediction_list) # for one iteration of the k-fold cross validation

	target_r2 = 1 - HP_SET_R2

	return {'loss': target_r2, 'status': STATUS_OK, 'model': model}

trials = Trials()

best = fmin(fn=optimiser,
			space=space,
			algo=tpe.suggest,
			trials=trials,
			max_evals=70,
			early_stop_fn=hyperopt.early_stop.no_progress_loss(50))

# print(best)

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']

print(best_model)