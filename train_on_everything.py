import pandas as pd
import random
import matplotlib.pyplot as plt
import periodictable
import xgboost as xg
from matrix_functions import make_test, make_train, range_setter
from sklearn.metrics import r2_score, mean_squared_error
import time

endfb8 = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
jendl5 = pd.read_csv('JENDL5_arange_all_features.csv')
tendl21 = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
jeff33 = pd.read_csv('JEFF33_features_arange_zeroed.csv')
cendl32 = pd.read_csv('CENDL33_features_arange_zeroed.csv')

all_libraries = pd.concat([endfb8, jendl5])
print(all_libraries.shape)
print(endfb8.shape)
print(jendl5.shape)

# all_libraries = pd.concat([all_libraries, tendl21])
all_libraries = pd.concat([all_libraries, jeff33])
all_libraries = pd.concat([all_libraries, cendl32])
print(all_libraries.shape)



nucs = range_setter(df=all_libraries, la=30, ua=210)
print(len(nucs))

all_libraries.index = range(len(all_libraries))


validation_nuclides = [[39,85], [39,86],[39,87],
					   [39,88],[39,90],[39,91],
					   [39,92],[39,93],[39,94],
					   [39,95]]
validation_set_size = 5 # number of nuclides hidden from training

while len(validation_nuclides) < validation_set_size:
	choice = random.choice(nucs) # randomly select nuclide from list of all nuclides
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


X_train, y_train = make_train(df=all_libraries, validation_nuclides=validation_nuclides, la=30, ua=215)
X_test, y_test = make_test(validation_nuclides, df=tendl21)

print("Data prep done")

model = xg.XGBRegressor(n_estimators=900,
                        learning_rate=0.01,
                        max_depth=8,
                        subsample=0.18236,
                        max_leaves=0,
                        seed=42, )

time1= time.time()

model.fit(X_train, y_train)

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
    plt.plot(erg, pred_xs, label='predictions')
    plt.plot(erg, true_xs, label='data')
    plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
    plt.legend()
    plt.grid()
    plt.ylabel('XS / b')
    plt.xlabel('Energy / MeV')
    plt.show()

    r2 = r2_score(true_xs, pred_xs)  # R^2 score for this specific nuclide
    print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")
    time.sleep(1.2)

print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}")  # MSE
print(f"R2: {r2_score(y_test, predictions)}")  # Total R^2 for all predictions in this training campaign
print(f'completed in {time.time() - time1} s')