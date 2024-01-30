import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
from matrix_functions import range_setter
from propagator_functions import training_sampler

ENDFBVIII = pd.read_csv('ENDFBVIII_MT16_uncertainties.csv')
ENDFB_nuclides = range_setter(df=ENDFBVIII, la=0, ua= 210)

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



target_nuclide = []
n_evaluations = 10

for i in range(n_evaluations):

	X_train, y_train = training_sampler(df=ENDFBVIII, LA=30, UA=210, sampled_uncertainties= ['unc_sn'])
