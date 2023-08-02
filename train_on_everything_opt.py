import pandas as pd
import random
import matplotlib.pyplot as plt
import periodictable
import xgboost as xg
from matrix_functions import make_test, make_train, range_setter, General_plotter
from sklearn.metrics import r2_score, mean_squared_error
import time
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import hyperopt.early_stop
import numpy as np
import scipy

endfb8 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
jendl5 = pd.read_csv('JENDL5_arange_all_features.csv')
tendl21 = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
jeff33 = pd.read_csv('JEFF33_features_arange_zeroed.csv')
cendl32 = pd.read_csv('CENDL33_features_arange_zeroed.csv')

all_libraries = pd.concat([endfb8, jendl5])
all_libraries = pd.concat([all_libraries, jeff33])
all_libraries = pd.concat([all_libraries, cendl32])


CENDL_nuclides = range_setter(df=cendl32, la=30, ua=210)
JENDL_nuclides = range_setter(df=jendl5, la=30, ua=210)
JEFF_nuclides = range_setter(df=jeff33, la=30, ua=210)
TENDL_nuclides = range_setter(df=tendl21, la=30, ua=210)
al = range_setter(df=endfb8, la=30, ua=210)

space = {'n_estimators': hp.choice('n_estimators', [500, 600, 700, 800, 850, 900, 1000, 1100,]),
         'subsample': hp.loguniform('subsample', np.log(0.1), np.log(1.0)),
         'max_leaves': 0,
         'max_depth': scope.int(hp.quniform("max_depth", 6, 12, 1)),
         'reg_lambda': hp.loguniform('lambda', np.log(0.5), np.log(3.0)),
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

    batch_r2 = []

    for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
        nuc = validation_nuclides[i]  # validation nuclide

        current_nuclide = nuc

        evaluation_r2s = []

        if current_nuclide in al:
            endfb8_erg, endfb8_xs = General_plotter(df=endfb8, nuclides=[current_nuclide])

            x_interpolate_endfb8 = np.linspace(start=0, stop=max(endfb8_erg), num=500)
            f_pred_endfb8 = scipy.interpolate.interp1d(x=erg, y=pred_xs)
            predictions_interpolated_endfb8 = f_pred_endfb8(x_interpolate_endfb8)

            f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg, y=endfb8_xs)
            endfb8_interp_xs = f_endfb8(x_interpolate_endfb8)
            pred_endfb_r2 = r2_score(y_true=endfb8_interp_xs, y_pred=predictions_interpolated_endfb8)
            evaluation_r2s.append(pred_endfb_r2)
        # print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MSE: {pred_endfb_mse:0.6f}")

        if current_nuclide in CENDL_nuclides:
            cendl_erg, cendl_xs = General_plotter(df=cendl32, nuclides=[current_nuclide])

            x_interpolate_cendl = np.linspace(start=0.000000001, stop=max(cendl_erg), num=500)
            f_pred_cendl = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
            predictions_interpolated_cendl = f_pred_cendl(x_interpolate_cendl)

            f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs)
            cendl_interp_xs = f_cendl32(x_interpolate_cendl)
            pred_cendl_r2 = r2_score(y_true=cendl_interp_xs, y_pred=predictions_interpolated_cendl)
            evaluation_r2s.append(pred_cendl_r2)
        # print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

        if current_nuclide in JENDL_nuclides:
            jendl_erg, jendl_xs = General_plotter(df=jendl5, nuclides=[current_nuclide])

            x_interpolate_jendl = np.linspace(start=0, stop=max(jendl_erg), num=500)
            f_pred_jendl = scipy.interpolate.interp1d(x=erg, y=pred_xs)
            predictions_interpolated_jendl = f_pred_jendl(x_interpolate_jendl)

            f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs)
            jendl_interp_xs = f_jendl5(x_interpolate_jendl)
            pred_jendl_r2 = r2_score(y_true=jendl_interp_xs, y_pred=predictions_interpolated_jendl)
            evaluation_r2s.append(pred_jendl_r2)
        # print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

        if current_nuclide in JEFF_nuclides:
            jeff_erg, jeff_xs = General_plotter(df=jeff33, nuclides=[current_nuclide])

            x_interpolate_jeff = np.linspace(start=0.0000000001, stop=max(jeff_erg), num=500)
            f_pred_jeff = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
            predictions_interpolated_jeff = f_pred_jeff(x_interpolate_jeff)

            f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs)
            jeff_interp_xs = f_jeff33(x_interpolate_jeff)
            pred_jeff_r2 = r2_score(y_true=jeff_interp_xs, y_pred=predictions_interpolated_jeff)
            evaluation_r2s.append(pred_jeff_r2)

        if current_nuclide in TENDL_nuclides:
            tendl_erg, tendl_xs = General_plotter(df=tendl21, nuclides=[current_nuclide])

            x_interpolate_tendl = np.linspace(start=0.00000000001, stop=max(tendl_erg), num=500)
            f_pred_tendl = scipy.interpolate.interp1d(x=erg, y=pred_xs, fill_value='extrapolate')
            predictions_interpolated_tendl = f_pred_tendl(x_interpolate_tendl)

            f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs)
            tendl_interp_xs = f_tendl21(x_interpolate_tendl)
            pred_tendl_r2 = r2_score(y_true=tendl_interp_xs, y_pred=predictions_interpolated_tendl)
            evaluation_r2s.append(pred_tendl_r2)
        # print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

        batch_r2.append(np.mean(evaluation_r2s))

    r2 = np.mean(batch_r2)

    print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}")  # MSE
    print(f"R2: {r2_score(y_test, predictions)}")  # Total R^2 for all predictions in this training campaign
    r2_return = 1 - r2

    return {'loss': r2_return, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=optimiser,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=100,
            early_stop_fn=hyperopt.early_stop.no_progress_loss(20))

print(best)


best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
print(best_model)