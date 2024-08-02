from MT_103_functions import make_train, make_test, range_setter
import matplotlib.pyplot as plt
import xgboost as xg
import pandas as pd


df = pd.read_csv('ENDFBVIII_MT16_MT103_MT107_fund_features.csv')