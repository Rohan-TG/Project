import datetime
import warnings
import scipy.stats
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import make_train, make_test,\
	range_setter, General_plotter
import scipy.stats
from datetime import timedelta
import tqdm

runtime = time.time()

TENDL21 = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
TENDL21.index = range(len(TENDL21))
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv") # training
df = df[df.Z != 11]
df.index = range(len(df))

TENDL_nuclides = range_setter(df=TENDL21, la=30, ua=215)

al = range_setter(la=30, ua=215, df=df)


n_evaluations = 50
datapoint_matrix = []
target_nuclide = [45,107]

validation_nuclides = [target_nuclide]
validation_set_size = 1  # train on all ENDF/B-VIII

X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=215, )  # make training matrix

X_test, y_test = make_test(validation_nuclides, df=TENDL21, )
print("\nTest nuclide selection complete")
print("Data prep done")
for i in tqdm.tqdm(range(n_evaluations)):
	print(f"\nRun {i+1}/{n_evaluations}")




	# while len(validation_nuclides) < validation_set_size:
	# 	choice = random.choice(al)  # randomly select nuclide from list of all nuclides
	# 	if choice not in validation_nuclides:
	# 		validation_nuclides.append(choice)


	time1 = time.time()

	model_seed = random.randint(a=1, b=1000) # seed for subsampling

	model = xg.XGBRegressor(n_estimators=900,
							learning_rate=0.008,
							max_depth=8,
							subsample=0.18236,
							max_leaves=0,
							seed=model_seed,)

	model.fit(X_train, y_train)

	print("Training complete")

	predictions = model.predict(X_test) # XS predictions
	predictions_ReLU = []
	for pred in predictions:
		if pred >= 0.003: # gate
			predictions_ReLU.append(pred)
		else:
			predictions_ReLU.append(0)

	predictions = predictions_ReLU

	if i == 0:
		for k, pred in enumerate(predictions):
			if [X_test[k,0], X_test[k,1]] == validation_nuclides[0]:
				datapoint_matrix.append([pred])
	else:
		valid_predictions = []
		for k, pred in enumerate(predictions):
			if [X_test[k, 0], X_test[k, 1]] == validation_nuclides[0]:
				valid_predictions.append(pred)
		for m, prediction in zip(datapoint_matrix, valid_predictions):
			m.append(prediction)

	print(f'completed in {time.time() - time1:0.1f} s')


XS_plot = []
E_plot = []

for i, row in enumerate(X_test):
	if [row[0], row[1]] == validation_nuclides[0]:
		XS_plot.append(y_test[i])
		E_plot.append(row[4]) # Energy values are in 5th row


datapoint_means = []
datapoint_upper_interval = []
datapoint_lower_interval = []

d_l_1sigma = []
d_u_1sigma = []

for point in datapoint_matrix:

	mu = np.mean(point)
	sigma = np.std(point)
	interval_low, interval_high = scipy.stats.t.interval(0.95, loc=mu, scale=sigma, df=(len(point) - 1))
	datapoint_means.append(mu)
	datapoint_upper_interval.append(interval_high)
	datapoint_lower_interval.append(interval_low)



lower_bound = []
upper_bound = []
for point, up, low, in zip(datapoint_means, datapoint_upper_interval, datapoint_lower_interval):
	lower_bound.append(point-low)
	upper_bound.append(point+up)





tendl_energy, tendl_xs = General_plotter(df=TENDL21, nuclides=[target_nuclide])



JEFF33 = pd.read_csv('JEFF33_all_features.csv')
JEFF33.index = range(len(JEFF33))
JEFF_nuclides = range_setter(df=JEFF33, la=30, ua=210)
JEFF_energy, JEFF_XS = General_plotter(df=JEFF33, nuclides=[target_nuclide])


JENDL5 = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL5.index = range(len(JENDL5))
JENDL_nuclides = range_setter(df=JENDL5, la=30, ua=210)
JENDL5_energy, JENDL5_XS = General_plotter(df=JENDL5, nuclides=[target_nuclide])


CENDL32 = pd.read_csv('CENDL32_all_features.csv')
CENDL32.index = range(len(CENDL32))
CENDL_nuclides = range_setter(df=CENDL32, la=30, ua=210)
CENDL32_energy, CENDL32_XS = General_plotter(df=CENDL32, nuclides=[target_nuclide])




# 2sigma CF
plt.plot(E_plot, datapoint_means, label = 'Prediction', color='red')
if target_nuclide in al:
	plt.plot(E_plot, XS_plot, label = 'ENDF/B-VIII', linewidth=2)
plt.plot(tendl_energy, tendl_xs, label = 'TENDL21', color='dimgrey', linewidth=2)
if target_nuclide in JEFF_nuclides:
	plt.plot(JEFF_energy, JEFF_XS, label='JEFF3.3', color='mediumvioletred')
if target_nuclide in JENDL_nuclides:
	plt.plot(JENDL5_energy, JENDL5_XS, '--', label='JENDL5', color='green')
if target_nuclide in CENDL_nuclides:
	plt.plot(CENDL32_energy, CENDL32_XS, '--', label = 'CENDL3.2', color='gold')
plt.fill_between(E_plot, datapoint_lower_interval, datapoint_upper_interval, alpha=0.2, label='95% CI', color='red')
plt.grid()
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[target_nuclide[0]]}-{target_nuclide[1]}")
plt.xlabel("Energy / MeV")
plt.ylabel("$\sigma_{n,2n}$ / b")
plt.legend(loc='upper left')
plt.show()


final_runtime = time.time() - runtime

print(f"Runtime: {timedelta(seconds=final_runtime)}")
