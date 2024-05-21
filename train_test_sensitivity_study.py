import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import xgboost as xg
import tqdm
import time
# import shap
from sklearn.metrics import mean_squared_error, r2_score
import periodictable
import scipy
from matrix_functions import make_train, make_test, range_setter, General_plotter

df = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")  # dataframe as above, but with the new features from the Gilbert-Cameron model
al = range_setter(df=df, la=0, ua=260)



thr_erg = []
for nuclide in tqdm.tqdm(al, total=len(al)):
	erg, xs = General_plotter(df=df, nuclides=[nuclide])

	for i,j in zip(erg,xs):
		if j > 0.0:
			thr_erg.append(i)
			break

aplot = [nuc[1] for nuc in al]

plt.figure()
plt.plot(aplot, thr_erg, 'x', label = 'nuclide')
plt.xlabel('Nucleon number (A)')
plt.ylabel('Threshold energy / MeV')
plt.title('Threshold energy by nuclide')
plt.grid()
plt.show()