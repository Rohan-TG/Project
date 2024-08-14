from matrix_functions import r2_standardiser
import pandas as pd
import keras
import random
import tensorflow
import time
from neural_matrix_functions import make_train, make_test, range_setter, General_plotter
import periodictable
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import mean_squared_error, r2_score

JENDL = pd.read_csv('JENDL5_with_MT16_and_ENDFB_MT103_107.csv')
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)

TENDL = pd.read_csv('TENDL_2021_MT16_with_ENDFB_MT103_107.csv')
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)

CENDL = pd.read_csv('CENDL32_MT16_with_ENDFB_MT103_107.csv')
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)

JEFF = pd.read_csv('JEFF33_all_features_MT16_103_107.csv')
JEFF_nuclides = range_setter(df=JEFF, la=30, ua = 210)

df = pd.read_csv('ENDFBVIII_MT16_91_103_107.csv')
ENDFB_nuclides = range_setter(df=df, la=30, ua=208)

validation_nuclides = [[40,90]]
validation_set_size = 20

while len(validation_nuclides) < validation_set_size: # up to 25 nuclides
	choice = random.choice(ENDFB_nuclides) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")


X_train, y_train = make_train(df=df, validation_nuclides= validation_nuclides, la=30, ua=208)

X_test, y_test = make_test(nuclides=validation_nuclides, df=df)


print("Matrices formed")

callback = keras.callbacks.EarlyStopping(monitor='loss',
										 min_delta=0.005,
										 patience=20,
										 mode='min',
										 start_from_epoch=20)

model = keras.Sequential()
model.add(keras.layers.Dense(100, input_shape=(17,), kernel_initializer='normal', activation='relu',
							 ))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(800, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1022, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear',))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train,
    y_train,
    epochs=100,
	batch_size= 100,
	callbacks=callback,
    validation_split = 0.2,
	verbose=1,)

predictions = model.predict(X_test)

predictions = predictions.ravel()

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
			dummy_test_E.append(row[2])  # Energy values are in 5th row
			dummy_predictions.append(predictions[i])

	XS_plotmatrix.append(dummy_test_XS)
	E_plotmatrix.append(dummy_test_E)
	P_plotmatrix.append(dummy_predictions)

for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
	all_preds = []
	all_libs = []
	current_nuclide = validation_nuclides[i]
	target_nuclide = current_nuclide

	interpolation_function = scipy.interpolate.interp1d(erg, y=pred_xs,
														fill_value='extrapolate')

	jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
	cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
	jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
	tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])

	nuc = validation_nuclides[i]  # validation nuclide
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
	plt.plot(tendlerg, tendlxs, label="TENDL-2021", color='dimgrey', linewidth=2)
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
	if nuc in JENDL_nuclides:
		plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
	if nuc in CENDL_nuclides:
		plt.plot(cendlerg, cendlxs, '--', label='CENDL-3.2', color='gold')
	plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
	plt.legend()
	plt.grid()
	plt.ylabel('$\sigma_{n,2n}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
	time.sleep(1.8)

	pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
	pred_endfb_gated, truncated_endfb, pred_endfb_r2 = r2_standardiser(predicted_xs=pred_xs, library_xs=true_xs)

	for x, y in zip(pred_endfb_gated, truncated_endfb):
		all_libs.append(y)
		all_preds.append(x)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f}")

	# Find r2 w.r.t. other libraries
	if target_nuclide in JENDL_nuclides:
		jendlxs_interpolated = interpolation_function(jendlerg)

		predjendlgated, gated_jendl_xs, jendl_r2 = r2_standardiser(library_xs=jendlxs,
																   predicted_xs=jendlxs_interpolated)
		# jendl_r2s.append(jendl_r2)
		jendl_rmse = mean_squared_error(predjendlgated, gated_jendl_xs) ** 0.5
		# jendl_rmses.append(jendl_rmse)

		for x, y in zip(gated_jendl_xs, predjendlgated):
			all_libs.append(x)
			all_preds.append(y)

	if target_nuclide in CENDL_nuclides:
		cendlxs_interpolated = interpolation_function(cendlerg)

		predcendlgated, truncatedcendl, cendl_r2 = r2_standardiser(library_xs=cendlxs,
																   predicted_xs=cendlxs_interpolated)
		# cendl_r2s.append(cendl_r2)
		cendl_rmse = mean_squared_error(predcendlgated, truncatedcendl) ** 0.5
		# cendl_rmses.append(cendl_rmse)

		for x, y in zip(truncatedcendl, predcendlgated):
			all_libs.append(x)
			all_preds.append(y)

	if target_nuclide in JEFF_nuclides:
		jeffxs_interpolated = interpolation_function(jefferg)

		predjeffgated, d2, jeff_r2 = r2_standardiser(library_xs=jeffxs,
													 predicted_xs=jeffxs_interpolated)
		# jeff_r2s.append(jeff_r2)
		jeff_rmse = mean_squared_error(d2, predjeffgated) ** 0.5
		# jeff_rmses.append(jeff_rmse)
		for x, y in zip(d2, predjeffgated):
			all_libs.append(x)
			all_preds.append(y)

	tendlxs_interpolated = interpolation_function(tendlerg)
	predtendlgated, truncatedtendl, tendlr2 = r2_standardiser(library_xs=tendlxs, predicted_xs=tendlxs_interpolated)
	# tendl_r2s.append(tendlr2)
	tendlrmse = mean_squared_error(truncatedtendl, predtendlgated) ** 0.5
	# tendl_rmses.append(tendlrmse)
	for x, y in zip(truncatedtendl, predtendlgated):
		all_libs.append(x)
		all_preds.append(y)

	print(f"Consensus R2: {r2_score(all_libs, all_preds):0.5f}")




# test_ERG, test_xs = General_plotter(df=df, nuclides=validation_nuclides)
# plt.plot(test_ERG, test_xs, label = 'ENDF/B-VIII')
# plt.plot(test_ERG, predictions, label = 'Prediction')
# plt.grid()
# plt.xlabel("Energy / MeV")
# plt.ylabel("XS / b")
# plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[validation_nuclides[0][0]]}-{validation_nuclides[0][1]}")
# plt.legend()
# plt.show()

plt.figure()
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.plot(history.history['loss'], label = 'Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()