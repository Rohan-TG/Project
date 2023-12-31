import random

import pandas as pd
import numpy as np
from matrix_functions import make_test, make_train, range_setter
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xg
import time
import datetime
import matplotlib.pyplot as plt

time1 = time.time()

ENDFBVIII = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
CENDL32 = pd.read_csv("CENDL32_all_features.csv")
JEFF33 = pd.read_csv("JEFF33_all_features.csv")
TENDL21 = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
JENDL5 = pd.read_csv("JENDL5_arange_all_features.csv")

print("Libraries loaded")

ENDFB_nuclides = range_setter(df=ENDFBVIII, la=30, ua=210)
TENDL21_nuclides = range_setter(df=TENDL21, la=30, ua=210)
JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=210)
CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=210)
JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=210)


all_libraries = pd.concat([ENDFBVIII, JENDL5])

# all_libraries = pd.concat([all_libraries, tendl21])
all_libraries = pd.concat([all_libraries, JEFF33])
all_libraries = pd.concat([all_libraries, CENDL32])
all_libraries.index = range(len(all_libraries))

all_libraries_nuclides = range_setter(df=all_libraries, la=30, ua=210)

exotic_TENDL_nuclides = []
# for tendl_nuclide in TENDL21_nuclides:
#     if tendl_nuclide not in all_libraries_nuclides:
#         exotic_TENDL_nuclides.append(tendl_nuclide)

for tendl_nuclide in TENDL21_nuclides:
    if tendl_nuclide not in ENDFB_nuclides:
        exotic_TENDL_nuclides.append(tendl_nuclide)

nuclides_used = []

mass_difference_with_r2 = []

print("Forming matrices...")

X_train, y_train = make_train(df=all_libraries, validation_nuclides=exotic_TENDL_nuclides, la=30, ua=210)  # make training matrix

X_test, y_test = make_test(exotic_TENDL_nuclides, df=TENDL21)

model_seed = random.randint(a=1, b=1000)

model = xg.XGBRegressor(n_estimators=900,
                        learning_rate=0.008,
                        max_depth=8,
                        subsample=0.18236,
                        max_leaves=0,
                        seed=model_seed, )

print("Training...")
model.fit(X_train, y_train)

print("Making predictions...")
predictions = model.predict(X_test)  # XS predictions
predictions_ReLU = []
for pred in predictions:
    if pred >= 0.003:
        predictions_ReLU.append(pred)
    else:
        predictions_ReLU.append(0)

predictions = predictions_ReLU

print("Calculating Differences...")
# Form arrays for plots below
XS_plotmatrix = []
E_plotmatrix = []
P_plotmatrix = []
for nuclide in exotic_TENDL_nuclides:
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

r2count = []

for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):

    target_exotic_nuclide = exotic_TENDL_nuclides[i]  # exotic nuclide

    r2 = r2_score(y_true=true_xs, y_pred=pred_xs)
    if r2 < 0.85 and r2 > -0.5:
        r2count.append(r2)

    match_element = []
    for nuclide in ENDFB_nuclides:
        if nuclide[0] == target_exotic_nuclide[0]:
            match_element.append(nuclide)
    nuc_differences = []
    for match_nuclide in match_element:
        difference = match_nuclide[1] - target_exotic_nuclide[1]
        nuc_differences.append(difference)

    if len(nuc_differences) > 0:
        min_difference = min(nuc_differences)

        mass_difference_with_r2.append([r2, min_difference])


log_r2_plots = [abs(np.log(abs(i[0]))) for i in mass_difference_with_r2]
mass_differences = [j[1] for j in mass_difference_with_r2]

plt.plot(mass_differences, log_r2_plots, 'x')
plt.grid()
plt.xlabel('Mass differences')
plt.ylabel("$|\ln(|r^2|)|$")
plt.show()

time2 = time.time()

timediff = time2 - time1

print(f"Good prediction count: {len(r2count)}/{len(exotic_TENDL_nuclides)}")
print(f"Time taken: {str(datetime.timedelta(seconds=timediff))}")
