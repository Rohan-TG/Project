import pandas as pd
import matplotlib.pyplot as plt
import math

# df = pd.read_csv('ENDFBVIII_100keV_all_uncertainties.csv')
#
# err = df['dXS'].values
# xs = df['XS'].values
#
# rel_err = []
# for dxs, x in zip(err, xs):
# 	if not math.isnan(dxs):
# 		rel_err.append((dxs/x)*100)
#
#
# plt.figure()
# plt.hist(err, bins=50)
# plt.grid()
# plt.show()

valnum = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 80, 100]

msescore = [0.026766 ,0.026649,0.026862, 0.026826, 0.026859, 0.026833, 0.026979,  0.027171, 0.027063, 0.026927, 0.027828,0.027745]
mseerr = [0.000273, 0.000267,0.000330, 0.000369, 0.000243, 0.000372, 0.000311,0.000463, 0.000424, 0.000261,0.000521, 0.000532]
