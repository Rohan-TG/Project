import pandas as pd
import matplotlib.pyplot as plt
import periodictable
from matrix_functions import General_plotter, range_setter, make_train, make_test, dsigma_dE, r2_standardiser
import random
import shap
import xgboost as xg
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import time

TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=0, ua=210)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=0, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=0, ua=210)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=0, ua=210)


df_test = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")


df_test.index = range(len(df_test)) # re-label indices
df.index = range(len(df))
# df_test = anomaly_remover(dfa = df_test)
al = range_setter(la=30, ua=210, df=df)

# tempal = [[16, 34], [66, 160], [50, 123], [48, 114], [44, 102], [52, 122], [18, 37], [52, 126], [44, 106],
# 		  [38, 85], [69, 171], [53, 131], [42, 96], [61, 148], [24, 51], [72, 174], [71, 176], [24, 54],
# 		  [54, 135], [20, 43], [48, 107], [47, 112], [36, 85], [29, 63], [65, 159], [78, 192], [53, 128],
# 		  [59, 141], [52, 131], [16, 36], [76, 185], [68, 167], [52, 120], [42, 99], [31, 70], [41, 95],
# 		  [17, 37], [54, 130], [70, 174], [56, 139], [44, 98], [58, 137], [54, 133], [52, 128], [54, 132],
# 		  [48, 106], [49, 113], [36, 82], [32, 70], [42, 92], [33, 74], [56, 133], [36, 83], [75, 185],
# 		  [77, 193], [50, 122], [48, 108], [38, 84], [72, 176], [53, 135], [38, 87], [52, 132], [20, 40],
# 		  [62, 144], [28, 58], [55, 136], [72, 175], [64, 157], [44, 99], [19, 41], [52, 130], [58, 141],
# 		  [48, 116], [40, 90], [15, 31], [64, 158], [50, 119], [74, 185], [14, 32], [25, 54], [16, 35],
# 		  [63, 154], [63, 155], [59, 143], [41, 94], [42, 100], [20, 45], [63, 153], [51, 126], [65, 158],
# 		  [17, 35], [22, 47], [58, 142], [40, 93], [36, 81], [45, 105], [60, 144], [52, 124], [76, 189],
# 		  [19, 39], [28, 63], [38, 89], [48, 112], [22, 49], [51, 125], [58, 143], [44, 104], [66, 158],
# 		  [32, 76], [14, 31], [65, 160], [16, 32], [81, 204], [81, 205], [50, 115], [66, 161], [58, 139],
# 		  [38, 86], [50, 120], [46, 108], [46, 106], [61, 149], [50, 126], [46, 103], [52, 121], [54, 129],
# 		  [38, 90], [24, 50], [48, 110], [40, 95], [36, 84], [54, 127], [54, 125], [62, 146], [61, 145], [50, 125],
# 		  [54, 134], [19, 40], [46, 109], [55, 135], [34, 77], [61, 146], [63, 152], [60, 150], [48, 111],
# 		  [63, 156], [34, 81], [66, 154], [50, 118], [61, 151], [71, 175], [46, 102], [30, 64], [66, 157],
# 		  [51, 122], [16, 33], [31, 69], [18, 38], [46, 105], [64, 160], [39, 91], [63, 157], [50, 124],
# 		  [34, 79], [54, 136], [61, 147], [56, 140], [53, 129], [82, 208], [18, 41], [31, 71], [36, 80],
# 		  [34, 82], [44, 97], [64, 155], [54, 123], [42, 93], [37, 87], [52, 125], [61, 144], [53, 132],
# 		  [46, 110], [53, 130], [50, 112], [44, 100], [48, 113], [62, 147], [58, 140], [50, 113], [68, 162],
# 		  [55, 137], [80, 197], [78, 193], [61, 150], [54, 124], [36, 86], [57, 139], [53, 133], [44, 103],
# 		  [44, 96], [36, 79], [43, 98], [51, 124], [57, 138], [49, 115], [46, 107], [58, 144], [23, 49],
# 		  [30, 65], [14, 30], [26, 54], [52, 123], [73, 182], [62, 148], [74, 181], [41, 93], [80, 201],
# 		  [68, 164]]

tempal = [[16, 35], [58, 137], [52, 126], [64, 155], [68, 167], [54, 135], [63, 157], [44, 96], [63, 152], [50, 113],
		  [52, 124], [80, 197], [36, 86], [14, 30], [16, 32], [81, 205], [58, 143], [20, 40], [33, 74], [30, 64],
		  [51, 122], [36, 79], [46, 107], [54, 124], [50, 119], [48, 116], [16, 33], [84, 209], [40, 93], [52, 120],
		  [44, 106], [65, 159], [63, 154], [61, 151], [54, 132], [55, 136], [50, 112], [41, 95], [78, 192], [34, 79],
		  [59, 143], [62, 146], [38, 87], [64, 157], [52, 131], [46, 109], [49, 113], [48, 108], [14, 31], [18, 38],
		  [56, 133], [54, 127], [16, 36], [28, 58], [62, 148], [44, 100], [18, 41], [17, 35], [70, 174], [55, 135],
		  [34, 81], [77, 193], [78, 193], [22, 47], [73, 182], [65, 160], [17, 37], [81, 204], [43, 98], [53, 131],
		  [58, 144], [54, 129], [31, 69], [20, 47], [38, 84], [66, 160], [52, 128], [60, 144], [38, 89], [36, 85],
		  [18, 37], [64, 158], [66, 157], [63, 155], [53, 130], [48, 114], [44, 99], [53, 128], [16, 34], [52, 130],
		  [31, 70], [48, 111], [50, 125], [52, 125], [63, 156], [54, 125], [61, 150], [36, 83], [34, 82], [54, 123],
		  [26, 54], [80, 201], [59, 141], [48, 112], [57, 139], [31, 71], [46, 102], [64, 160], [51, 124], [58, 139],
		  [28, 63], [74, 181], [18, 40], [42, 96], [68, 164], [58, 141], [65, 158], [58, 140], [24, 51], [29, 63],
		  [14, 32], [57, 138], [47, 112], [19, 40], [19, 41], [61, 145], [23, 49], [51, 126], [50, 123], [52, 121],
		  [30, 65], [44, 102], [22, 49], [61, 146], [69, 171], [62, 147], [41, 94], [48, 106], [63, 153], [20, 45],
		  [53, 129], [50, 118], [24, 50], [44, 105], [56, 140], [58, 142], [68, 162], [26, 57], [24, 54], [46, 110],
		  [44, 103], [54, 136], [75, 185], [44, 97], [56, 139], [71, 176], [44, 104], [50, 120], [45, 105], [40, 95],
		  [61, 147], [54, 133], [52, 122], [42, 99], [55, 137], [71, 175], [72, 174], [52, 123], [54, 130], [34, 77],
		  [48, 113], [76, 189], [46, 108], [46, 106], [48, 110], [39, 91], [61, 149], [25, 54], [37, 87], [38, 86],
		  [53, 133], [40, 90], [53, 132], [44, 98], [20, 43], [42, 92], [41, 93], [50, 124], [49, 115], [38, 85],
		  [38, 90], [51, 125], [54, 134], [36, 84]]

# random.seed(a=10)

validation_nuclides = []
validation_set_size = 20

while len(validation_nuclides) < validation_set_size: # up to 25 nuclides
	choice = random.choice(tempal) # randomly select nuclide from list of all nuclides in ENDF/B-VIII
	if choice not in validation_nuclides:
		validation_nuclides.append(choice)
print("Test nuclide selection complete")



X_train, y_train = make_train(df=df, validation_nuclides=validation_nuclides, la=30, ua=210,) # create training matrix
X_test, y_test = make_test(validation_nuclides, df=df_test,) # create test matrix using validation nuclides
print("Data prep done")

model = xg.XGBRegressor(n_estimators=900, # define regressor
						learning_rate=0.008,
						max_depth=8,
						subsample=0.18236,
						max_leaves=0,
						seed=42, )

model.fit(X_train, y_train)
print("Training complete")
predictions = model.predict(X_test)  # XS predictions
predictions_ReLU = []

for pred in predictions: # prediction gate
	if pred >= 0.003:
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
	all_preds = []
	all_libs = []
	current_nuclide = validation_nuclides[i]

	jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
	cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
	jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
	tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])

	nuc = validation_nuclides[i]  # validation nuclide
	plt.plot(erg, pred_xs, label='Predictions', color='red')
	plt.plot(erg, true_xs, label='ENDF/B-VIII', linewidth=2)
	plt.plot(tendlerg, tendlxs, label="TENDL21", color='dimgrey', linewidth=2)
	if nuc in JEFF_nuclides:
		plt.plot(jefferg, jeffxs, '--', label='JEFF3.3', color='mediumvioletred')
	if nuc in JENDL_nuclides:
		plt.plot(jendlerg, jendlxs, label='JENDL5', color='green')
	if nuc in CENDL_nuclides:
		plt.plot(cendlerg, cendlxs, '--', label='CENDL3.2', color='gold')
	plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
	plt.legend()
	plt.grid()
	plt.ylabel('$\sigma_{n,2n}$ / b')
	plt.xlabel('Energy / MeV')
	plt.show()

	mse = mean_squared_error(y_true=true_xs, y_pred=pred_xs)
	print(f"\n{periodictable.elements[nuc[0]]}-{nuc[1]:0.0f}")
	time.sleep(0.8)

	pred_endfb_mse = mean_squared_error(pred_xs, true_xs)
	pred_endfb_gated, truncated_endfb, pred_endfb_r2 = r2_standardiser(raw_predictions=pred_xs, library_xs=true_xs)

	endfbmape = mean_absolute_percentage_error(truncated_endfb, pred_endfb_gated)
	for x, y in zip(pred_endfb_gated, truncated_endfb):
		all_libs.append(y)
		all_preds.append(x)
	print(f"Predictions - ENDF/B-VIII R2: {pred_endfb_r2:0.5f} MAPE: {endfbmape:0.4f}")

	# Find r2 w.r.t. other libraries
	if current_nuclide in CENDL_nuclides:

		cendl_test, cendl_xs = make_test(nuclides=[current_nuclide], df=CENDL)
		pred_cendl = model.predict(cendl_test)

		pred_cendl_gated, truncated_cendl, pred_cendl_r2 = r2_standardiser(raw_predictions=pred_cendl, library_xs=cendl_xs)

		pred_cendl_mse = mean_squared_error(pred_cendl, cendl_xs)
		for x, y in zip(pred_cendl_gated, truncated_cendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - CENDL3.2 R2: {pred_cendl_r2:0.5f} MSE: {pred_cendl_mse:0.6f}")

	if current_nuclide in JENDL_nuclides:

		jendl_test, jendl_xs = make_test(nuclides=[current_nuclide], df=JENDL)
		pred_jendl = model.predict(jendl_test)

		pred_jendl_mse = mean_squared_error(pred_jendl, jendl_xs)
		pred_jendl_gated, truncated_jendl, pred_jendl_r2 = r2_standardiser(raw_predictions=pred_jendl, library_xs=jendl_xs)

		for x, y in zip(pred_jendl_gated, truncated_jendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JENDL5 R2: {pred_jendl_r2:0.5f} MSE: {pred_jendl_mse:0.6f}")

	if current_nuclide in JEFF_nuclides:
		jeff_test, jeff_xs = make_test(nuclides=[current_nuclide], df=JEFF)

		pred_jeff = model.predict(jeff_test)

		pred_jeff_mse = mean_squared_error(pred_jeff, jeff_xs)
		pred_jeff_gated, truncated_jeff, pred_jeff_r2 = r2_standardiser(raw_predictions=pred_jeff, library_xs=jeff_xs)
		for x, y in zip(pred_jeff_gated, truncated_jeff):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - JEFF3.3 R2: {pred_jeff_r2:0.5f} MSE: {pred_jeff_mse:0.6f}")

	if current_nuclide in TENDL_nuclides:

		tendl_test, tendl_xs = make_test(nuclides=[current_nuclide], df=TENDL)
		pred_tendl = model.predict(tendl_test)

		pred_tendl_mse = mean_squared_error(pred_tendl, tendl_xs)
		pred_tendl_gated, truncated_tendl, pred_tendl_r2 = r2_standardiser(raw_predictions=pred_tendl, library_xs=tendl_xs)
		for x, y in zip(pred_tendl_gated, truncated_tendl):
			all_libs.append(y)
			all_preds.append(x)
		print(f"Predictions - TENDL21 R2: {pred_tendl_r2:0.5f} MSE: {pred_tendl_mse:0.6f}")

	print(f"Consensus R2: {r2_score(all_libs, all_preds):0.5f}")

	print(f"Turning points: {dsigma_dE(XS=pred_xs)}")
	# print(f"Mean gradient: {sum(abs(np.gradient(pred_xs)))/(len(pred_xs)):0.5f}")



# FIA
model.get_booster().feature_names = ['Z',
									 'A',
									 'S2n',
									 'S2p',
									 'E',
									 'Sp',
									 'Sn',
									 'BEA',
									 # 'P',
									 'Snc',
									 'g-def',
									 'N',
									 'b-def',
									 'Sn da',
									 # 'Sp d',
									 'S2n d',
									 'Radius',
									 'n_g_erg',
									 'n_c_erg',
									 'n_rms_r',
									 # 'oct_def',
									 # 'D_c',
									 'BEA_d',
									 'BEA_c',
									 # 'Pair_d',
									 # 'Par_d',
									 # 'S2n_c',
									 'S2p_c',
									 'ME',
									 # 'Z_even',
									 # 'A_even',
									 # 'N_even',
									 'Shell',
									 # 'Parity',
									 # 'Spin',
									 'Decay',
									 # 'Deform',
									 'p_g_e',
									 'p_c_e',
									 # 'p_rms_r',
									 # 'rms_r',
									 # 'Sp_c',
									 # 'Sn_c',
									 'Shell_c',
									 # 'S2p-d',
									 # 'Shell-d',
									 'Spin-c',
									 # 'Rad-c',
									 'Def-c',
									 # 'ME-c',
									 'BEA-A-c',
									 'Decay-d',
									 # 'ME-d',
									 # 'Rad-d',
									 # 'Pair-c',
									 # 'Par-c',
									 'BEA-A-d',
									 # 'Spin-d',
									 'Def-d',
									 # 'mag_p',
									 # 'mag-n',
									 # 'mag-d',
									 'Nlow',
									 # 'Ulow',
									 # 'Ntop',
									 'Utop',
									 # 'ainf',
									 'Asym',
									 # 'Asym_c',
									 'Asym_d',
									 # 'AM'
									 ]

# Gain-based feature importance plot
plt.figure(figsize=(10, 12))
xg.plot_importance(model, ax=plt.gca(), importance_type='total_gain', max_num_features=60)  # metric is total gain
plt.show()

# Splits-based feature importance plot
plt.figure(figsize=(10, 12))
plt.title("Splits-based FI")
xg.plot_importance(model, ax=plt.gca(), max_num_features=60)
plt.show()

explainer = shap.Explainer(model.predict, X_train,
						   feature_names= ['Z',
									 'A',
									 'S2n',
									 'S2p',
									 'E',
									 'Sp',
									 'Sn',
									 'BEA',
									 # 'P',
									 'Snc',
									 'g-def',
									 'N',
									 'b-def',
									 'Sn da',
									 # 'Sp d',
									 'S2n d',
									 'Radius',
									 'n_g_erg',
									 'n_c_erg',
									 'n_rms_r',
									 # 'oct_def',
									 # 'D_c',
									 'BEA_d',
									 'BEA_c',
									 # 'Pair_d',
									 # 'Par_d',
									 # 'S2n_c',
									 'S2p_c',
									 'ME',
									 # 'Z_even',
									 # 'A_even',
									 # 'N_even',
									 'Shell',
									 # 'Parity',
									 # 'Spin',
									 'Decay',
									 # 'Deform',
									 'p_g_e',
									 'p_c_e',
									 # 'p_rms_r',
									 # 'rms_r',
									 # 'Sp_c',
									 # 'Sn_c',
									 'Shell_c',
									 # 'S2p-d',
									 # 'Shell-d',
									 'Spin-c',
									 # 'Rad-c',
									 'Def-c',
									 # 'ME-c',
									 'BEA-A-c',
									 'Decay-d',
									 # 'ME-d',
									 # 'Rad-d',
									 # 'Pair-c',
									 # 'Par-c',
									 'BEA-A-d',
									 # 'Spin-d',
									 'Def-d',
									 # 'mag_p',
									 # 'mag-n',
									 # 'mag-d',
									 'Nlow',
									 # 'Ulow',
									 # 'Ntop',
									 'Utop',
									 # 'ainf',
									 'Asym',
									 # 'Asym_c',
									 'Asym_d',
									 # 'AM'
									 ]) # SHAP feature importance analysis
shap_values = explainer(X_test)
#
#
#
shap.plots.bar(shap_values, max_display = 70) # display SHAP results
shap.plots.waterfall(shap_values[0], max_display=70)
