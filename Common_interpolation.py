import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import periodictable
from sklearn.metrics import mean_squared_error
import scipy
TENDL = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
ENDFBVIII = pd.read_csv('ENDFBVIII_MT16_XS_feateng.csv')
JENDL5 = pd.read_csv('JENDL5_arange_all_features.csv')
JEFF33 = pd.read_csv('JEFF33_features_arange_zeroed.csv')
from matrix_functions import General_plotter

target_nuclide = [31,69]

endfb8_erg, endfb8_xs = General_plotter(df=ENDFBVIII, nuclides=[target_nuclide])
tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[target_nuclide])
jendl_erg, jendl_xs = General_plotter(df=JENDL5, nuclides=[target_nuclide])
jeff_erg, jeff_xs = General_plotter(df=JEFF33, nuclides=[target_nuclide])

plt.plot(endfb8_erg[0], endfb8_xs, label = 'ENDF/B-VIII')
plt.plot(tendl_erg[0], tendl_xs, label='TENDL21')
plt.plot(jendl_erg[0], jendl_xs, label = 'JENDL5')
plt.plot(jeff_erg[0], jeff_xs, label='JEFF3.3')
plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[target_nuclide[0]]}-{target_nuclide[1]}")
plt.xlabel('Energy / MeV')
plt.ylabel('$\sigma_{n,2n}$ / b')
plt.legend()
plt.grid()
plt.show()
f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg[0], y=endfb8_xs)
f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg[0], y=tendl_xs)
f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg[0], y=jendl_xs)
f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg[0], y=jeff_xs)

x_interpolate = np.linspace(stop=19.0, start=0, num=500)
endfb8_interp = f_endfb8(x_interpolate)
tendl_interp_xs = f_tendl21(x_interpolate)
jendl_interp_xs = f_jendl5(x_interpolate)
jeff_interp_xs = f_jeff33(x_interpolate)

rmse_endf_tendl = mean_squared_error(endfb8_interp, tendl_interp_xs)
rmse_endf_jendl = mean_squared_error(endfb8_interp, jendl_interp_xs)
rmse_tendl_jendl = mean_squared_error(jendl_interp_xs, tendl_interp_xs)
rmse_tendl_jeff = mean_squared_error(jeff_interp_xs, tendl_interp_xs)
rmse_endfb_jeff = mean_squared_error(endfb8_interp, jeff_interp_xs)
rmse_jeff_jendl = mean_squared_error(jeff_interp_xs, jendl_interp_xs)

print(f"ENDF/B-VIII - TENDL21: {rmse_endf_tendl:0.6f}")
print(f"ENDF/B-VIII - JENDL5: {rmse_endf_jendl:0.6f}")
print(f"TENDL21 - JENDL5: {rmse_tendl_jendl:0.6f}")
print(f"TENDL21 - JEFF3.3: {rmse_tendl_jeff:0.6f}")
print(f"ENDF/B-VIII - JEFF3.3: {rmse_endfb_jeff:0.6f}")
print(f"JEFF3.3 - JENDL5: {rmse_jeff_jendl:0.6f}")