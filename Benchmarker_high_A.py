import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import time
# import shap
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
from matrix_functions import anomaly_remover, make_train, make_test, range_setter

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")  # dataframe as above, but with the new features from the Gilbert-Cameron model




df_test = anomaly_remover(dfa = df_test)

# for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
# 	if [j, i] in al or i > 215 or i < 28:
# 		continue
# 	else:
# 		al.append([j, i]) # format is [Z, A]



al = range_setter(df=df, la=0, ua=270)


nuclides_used = []
nuclide_mse = []
nuclide_r2 = []

if __name__ == "__main__":
	# lower_a = int(input("Enter lower boundary A: "))
	# upper_a = int(input("Enter upper boundary A: "))

	# overall_r2_list = []

	every_prediction_list = []
	every_true_value_list = []

	counter = 0
	while len(nuclides_used) < len(al):


		# print(f"{len(nuclides_used) // len(al)} Epochs left")

		validation_nuclides = []  # list of nuclides used for validation
		# test_nuclides = []
		validation_set_size = 20  # number of nuclides hidden from training

		while len(validation_nuclides) < validation_set_size:

			choice = random.choice(al)  # randomly select nuclide from list of all nuclides
			if choice not in validation_nuclides and choice not in nuclides_used:
				validation_nuclides.append(choice)
				nuclides_used.append(choice)
			if len(nuclides_used) == len(al):
				break


		print("Test nuclide selection complete")
		print(f"{len(nuclides_used)}/{len(al)} selections")
		# print(f"Epoch {len(al) // len(nuclides_used) + 1}/")


		X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=0, ua=270) # make training matrix

		X_test, y_test = make_test(validation_nuclides, df=df_test)

		# X_train must be in the shape (n_samples, n_features)
		# and y_train must be in the shape (n_samples) of the target

		print("Train/val matrices generated")


		model = xg.XGBRegressor(n_estimators=900,
								learning_rate=0.01555,
								max_depth=8,
								subsample=0.18236,
								max_leaves=0,
								seed=42,)

		time1 = time.time()
		model.fit(X_train, y_train)

		print("Training complete")

		predictions = model.predict(X_test) # XS predictions


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
					dummy_test_E.append(row[4]) # Energy values are in 5th row
					dummy_predictions.append(predictions[i])

			XS_plotmatrix.append(dummy_test_XS)
			E_plotmatrix.append(dummy_test_E)
			P_plotmatrix.append(dummy_predictions)

		# plot predictions against data
		for i, (pred_xs, true_xs, erg) in enumerate(zip(P_plotmatrix, XS_plotmatrix, E_plotmatrix)):
			nuc = validation_nuclides[i]
			# plt.plot(erg, pred_xs, label='predictions')
			# plt.plot(erg, true_xs, label='data')
			# plt.title(f"(n,2n) XS for {periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
			# plt.legend()
			# plt.grid()
			# plt.ylabel('XS / b')
			# plt.xlabel('Energy / MeV')
			# plt.show()
			r2 = r2_score(true_xs, pred_xs) # R^2 score for this specific nuclide
			print(f"{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f} R2: {r2:0.5f}")

			mse = mean_squared_error(true_xs, pred_xs)

			nuclide_mse.append([nuc[0], nuc[1], mse])
			nuclide_r2.append([nuc[0], nuc[1], r2])
			# individual_r2_list.append(r2)

		overall_r2 = r2_score(y_test, predictions)
		for pred in predictions:
			every_prediction_list.append(pred)

		for val in y_test:
			every_true_value_list.append(val)
		# overall_r2_list.append(overall_r2)
		# print(f"MSE: {mean_squared_error(y_test, predictions, squared=False)}") # MSE
		print(f"R2: {overall_r2}") # Total R^2 for all predictions in this training campaign
		time_taken = time.time() - time1
		print(f'completed in {time_taken} s.\n')
		# time.sleep(10)



print()

print(f"New overall r2: {r2_score(every_true_value_list, every_prediction_list)}")

A_plots = [i[1] for i in nuclide_r2]
Z_plots = [i[0] for i in nuclide_r2]
# mse_log_plots = [np.log(i[-1]) for i in nuclide_mse]
# mse_plots = [i[-1] for i in nuclide_mse]

log_plots = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]
log_plots_Z = [abs(np.log(abs(i[-1]))) for i in nuclide_r2]

heavy_performance = [abs(np.log(abs(i[-1]))) for i in nuclide_r2 if i[1] < 50]
print(f"Heavy performance: {heavy_performance}, mean: {np.mean(heavy_performance)}")

plt.figure()
plt.plot(A_plots, log_plots, 'x')
plt.xlabel("A")
plt.ylabel("$|\ln(|r^2|)|$")
plt.yticks(rotation=90)
plt.title("Performance - A")
plt.grid()
plt.show()

plt.figure()
plt.plot(Z_plots, log_plots_Z, 'x')
plt.xlabel('Z')
plt.ylabel("$|\ln(|r^2|)|$")
plt.title("Performance - Z")
plt.grid()
plt.show()