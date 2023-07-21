import pandas as pd
import random
import matplotlib.pyplot as plt
import periodictable
import xgboost as xg
from matrix_functions import make_test, make_train, range_setter
from sklearn.metrics import r2_score, mean_squared_error
import time
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop
import numpy as np

endfb8 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
jendl5 = pd.read_csv('JENDL5_arange_all_features.csv')
tendl21 = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
jeff33 = pd.read_csv('JEFF33_features_arange_zeroed.csv')
cendl32 = pd.read_csv('CENDL33_features_arange_zeroed.csv')

all_libraries = pd.concat([endfb8, jendl5])
all_libraries = pd.concat([all_libraries, jeff33])
all_libraries = pd.concat([all_libraries, cendl32])

space = {'n_estimators': hp.choice('n_estimators', [400, 500, 600, 700, 800, 850, 900, 1000, 1100,
														]),
			 'subsample': hp.loguniform('subsample', np.log(0.1), np.log(1.0)),
			 'max_leaves': 0,
			 'max_depth': scope.int(hp.quniform("max_depth", 6, 12, 1)),
			 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.15))}

nucs = range_setter(df=all_libraries, la=30, ua=210)


all_libraries.index = range(len(all_libraries))

def optimiser(space):

    validation_nuclides = []
    validation_set_size = 30 # number of nuclides hidden from training

    while len(validation_nuclides) < validation_set_size:
        choice = random.choice(nucs) # randomly select nuclide from list of all nuclides
        if choice not in validation_nuclides:
            validation_nuclides.append(choice)
    print("Test nuclide selection complete")


    X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides, la=30, ua=215)
    X_test, y_test = make_test(validation_nuclides, df=tendl21)

    print("Data prep done")

    model = xg.XGBRegressor(**space, seed=42)

    evaluation = [(X_test, y_test)]

    model.fit(X_train, y_train, eval_set= evaluation, eval_metric='rmse')

    print("Training complete")

    predictions = model.predict(X_test)  # XS predictions
    predictions_ReLU = []
    for pred in predictions:
        if pred >= 0.001:
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
        nuc = validation_nuclides[i]  # validation nuclide
        # plt.plot(erg, pred_xs, label='predictions')
        # plt.plot(erg, true_xs, label='data')
        # plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
        # plt.legend()
        # plt.grid()
        # plt.ylabel('XS / b')
        # plt.xlabel('Energy / MeV')
        # plt.show()

        r2 = r2_score(true_xs, pred_xs)  # R^2 score for this specific nuclide
        print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
        # time.sleep(1.2)

    print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}")  # MSE
    print(f"R2: {r2_score(y_test, predictions)}")  # Total R^2 for all predictions in this training campaign
    r2_return = 1 - r2_score(y_test, predictions)

    return {'loss': r2_return, 'status': STATUS_OK, 'model': model}


trials = Trials()
best = fmin(fn=optimiser,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=70,
            early_stop_fn=hyperopt.early_stop.no_progress_loss(15))

print(best)


best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
print(best_model)