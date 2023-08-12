import pandas as pd
from matrix_functions import range_setter

tendl = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')


tendl = tendl[tendl.Z == 45]

tendl = tendl[tendl.A == 93]

print(tendl)