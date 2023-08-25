import pandas as pd
from matrix_functions import range_setter, General_plotter
from sklearn.metrics import mean_squared_error, r2_score
import random
import matplotlib.pyplot as plt
import scipy
import periodictable
import numpy as np
import time


TENDL = pd.read_csv("TENDL21_MT16_XS_features_zeroed.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=215, ua=270)

JEFF = pd.read_csv('JEFF33_features_arange_zeroed.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=215, ua=270)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=215, ua=270)

CENDL = pd.read_csv('CENDL32_features_arange_zeroed.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=215, ua=270)

ENDFB_VIII = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
ENDFB_VIII.index = range(len(ENDFB_VIII))
ENDFB_VIII_nuclides = range_setter(df=ENDFB_VIII, la=215, ua=270)


nucs = []
for i in range(20):
    nucs.append(random.choice(ENDFB_VIII_nuclides))




for i in nucs:

    current_nuclide = i

    endfberg, endfbxs = General_plotter(df=ENDFB_VIII, nuclides=[current_nuclide])
    jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
    cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
    jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
    tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])

    nuc = i  # validation nuclide
    plt.plot(endfberg, endfbxs, label='ENDF/B-VIII', linewidth=2)
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

    time.sleep(0.8)

    x_interpolate = np.linspace(start=0.0, stop=19.4, num=200)


    print(f"{periodictable.elements[nuc[0]]}-{nuc[1]}")

    if current_nuclide in nucs:
        endfb8_erg, endfb8_xs = General_plotter(df=ENDFB_VIII, nuclides=[current_nuclide])

        f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg, y=endfb8_xs)
        endfb8_interp_xs = f_endfb8(x_interpolate)

    if current_nuclide in CENDL_nuclides:
        cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[current_nuclide])


        f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs)
        cendl_interp_xs = f_cendl32(x_interpolate)

    if current_nuclide in TENDL_nuclides:
        tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[current_nuclide])

        f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs)
        tendl_interp_xs = f_tendl21(x_interpolate)

    if current_nuclide in JEFF_nuclides:
        jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[current_nuclide])

        f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs)
        jeff_interp_xs = f_jeff33(x_interpolate)

        rmse_tendl_jeff = r2_score(jeff_interp_xs, tendl_interp_xs)
        rmse_endfb_jeff = r2_score(endfb8_interp_xs, jeff_interp_xs)

        print(f"TENDL21 - JEFF3.3: {rmse_tendl_jeff:0.6f}")
        print(f"ENDF/B-VIII - JEFF3.3: {rmse_endfb_jeff:0.6f}")

    if current_nuclide in JENDL_nuclides:
        jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[current_nuclide])
        if max(jendl_erg) > 18:
            f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs)
            jendl_interp_xs = f_jendl5(x_interpolate)

            rmse_endf_jendl = r2_score(endfb8_interp_xs, jendl_interp_xs)
            rmse_tendl_jendl = r2_score(jendl_interp_xs, tendl_interp_xs)
            rmse_jeff_jendl = r2_score(jeff_interp_xs, jendl_interp_xs)

            print(f"ENDF/B-VIII - JENDL5: {rmse_endf_jendl:0.6f}")
            if nuc in JEFF_nuclides:
                print(f"JEFF3.3 - JENDL5: {rmse_jeff_jendl:0.6f}")
            print(f"TENDL21 - JENDL5: {rmse_tendl_jendl:0.6f}")



    rmse_endf_tendl = r2_score(endfb8_interp_xs, tendl_interp_xs)

    print(f"ENDF/B-VIII - TENDL21: {rmse_endf_tendl:0.6f}")
    print()
