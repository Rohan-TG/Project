import pandas as pd
from matrix_functions import range_setter, General_plotter
from sklearn.metrics import mean_squared_error, r2_score
import random
import matplotlib.pyplot as plt
import scipy
import periodictable
import numpy as np
import time


TENDL = pd.read_csv("TENDL_2021_MT16_XS_features.csv")
TENDL.index = range(len(TENDL))
TENDL_nuclides = range_setter(df=TENDL, la=30, ua=210)

JEFF = pd.read_csv('JEFF33_all_features.csv')
JEFF.index = range(len(JEFF))
JEFF_nuclides = range_setter(df=JEFF, la=30, ua=210)

JENDL = pd.read_csv('JENDL5_arange_all_features.csv')
JENDL.index = range(len(JENDL))
JENDL_nuclides = range_setter(df=JENDL, la=30, ua=210)

CENDL = pd.read_csv('CENDL32_all_features.csv')
CENDL.index = range(len(CENDL))
CENDL_nuclides = range_setter(df=CENDL, la=30, ua=210)

ENDFB_VIII = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
ENDFB_VIII.index = range(len(ENDFB_VIII))
ENDFB_VIII_nuclides = range_setter(df=ENDFB_VIII, la=30, ua=210)


# nucs = [[76, 191], [46, 102], [22, 47], [65, 159], [52, 122], [40, 92], [22, 49], [18, 39],
#         [68, 166], [61, 149], [17, 36], [66, 157], [38, 86], [38, 90], [35, 81], [49, 115],
#         [77, 191], [72, 176], [44, 99], [58, 139], [39, 89], [20, 41], [50, 115], [40, 96],
#         [78, 198], [61, 150], [74, 185], [50, 125], [66, 162], [50, 124], [47, 111], [35, 79],
#         [60, 149], [39, 90], [64, 160], [38, 87], [39, 91], [63, 152], [76, 189], [23, 51],
#         [75, 185], [52, 125], [42, 99], [41, 94], [69, 171], [68, 169], [19, 40], [58, 143],
#         [48, 109], [36, 86], [56, 139], [34, 77], [52, 126], [24, 54], [71, 175], [34, 79],
#         [44, 98], [78, 194], [70, 175], [50, 117], [44, 96], [55, 136], [81, 204], [58, 144],
#         [23, 49], [52, 123], [63, 156], [18, 38], [80, 203], [56, 132], [69, 170], [57, 140],
#         [52, 121], [52, 128], [59, 142], [50, 118], [84, 208], [57, 138], [66, 156], [78, 196],
#         [70, 176], [50, 123], [65, 161], [52, 124], [49, 113], [48, 114], [50, 119], [38, 85],
#         [37, 87], [51, 122], [54, 124], [40, 95], [19, 41], [54, 127], [44, 106], [54, 135],
#         [32, 75], [29, 64], [78, 192], [14, 31], [81, 205], [65, 158], [36, 85], [71, 176],
#         [68, 168], [18, 41], [72, 175], [50, 126], [50, 122], [73, 182], [74, 181], [20, 45],
#         [51, 125], [80, 202], [48, 110], [31, 69], [53, 133], [51, 124], [66, 163], [34, 82],
#         [42, 98], [56, 138], [54, 129], [70, 169], [41, 95], [46, 109], [36, 80], [20, 44],
#         [84, 209], [27, 58], [37, 86], [29, 63], [80, 196], [56, 140], [48, 111], [20, 42],
#         [80, 204], [64, 159], [53, 129], [78, 193], [77, 193], [46, 103], [54, 133], [68, 167],
#         [68, 170], [52, 120], [16, 35]]

nucs = [[22, 47], [65, 159], [52, 122], [17, 36], [66, 157], [38, 86], [38, 90], [61, 150], [74, 185], [50, 125], [50, 124], [35, 79], [60, 149], [39, 90], [64, 160], [38, 87], [39, 91], [63, 152], [52, 125], [41, 94], [69, 171], [68, 169], [19, 40], [56, 139], [34, 77], [52, 126], [71, 175], [34, 79], [70, 175], [50, 117], [55, 136], [23, 49], [63, 156], [57, 140], [52, 121], [52, 128], [59, 142], [50, 118], [50, 123], [65, 161], [52, 124], [50, 119], [38, 85], [51, 122], [19, 41], [54, 135], [32, 75], [81, 205], [71, 176], [72, 175], [50, 126], [50, 122], [51, 125], [53, 133], [34, 82], [70, 169], [41, 95], [46, 109], [84, 209], [29, 63], [56, 140], [20, 42], [64, 159], [54, 133], [68, 167], [16, 35]]

set_size = 1


outliers = []
false_positives = []

for i in nucs:

    current_nuclide = i
    auxilliary_library_comparisons = []
    endfb_comparisons = []

    endfberg, endfbxs = General_plotter(df=ENDFB_VIII, nuclides=[current_nuclide])
    jendlerg, jendlxs = General_plotter(df=JENDL, nuclides=[current_nuclide])
    cendlerg, cendlxs = General_plotter(df=CENDL, nuclides=[current_nuclide])
    jefferg, jeffxs = General_plotter(df=JEFF, nuclides=[current_nuclide])
    tendlerg, tendlxs = General_plotter(df=TENDL, nuclides=[current_nuclide])

    nuc = i  # validation nuclide
    plt.plot(endfberg, endfbxs, label='ENDF/B-VIII', linewidth=2)
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

    time.sleep(3)

    x_interpolate = np.linspace(start=0.0, stop=19.4, num=200)


    print(f"{periodictable.elements[nuc[0]]}-{nuc[1]}")

    if current_nuclide in nucs:
        endfb8_erg, endfb8_xs = General_plotter(df=ENDFB_VIII, nuclides=[current_nuclide])

        f_endfb8 = scipy.interpolate.interp1d(x=endfb8_erg, y=endfb8_xs, fill_value='extrapolate')
        endfb8_interp_xs = f_endfb8(x_interpolate)


    if current_nuclide in TENDL_nuclides:
        tendl_erg, tendl_xs = General_plotter(df=TENDL, nuclides=[current_nuclide])

        f_tendl21 = scipy.interpolate.interp1d(x=tendl_erg, y=tendl_xs, fill_value='extrapolate')
        tendl_interp_xs = f_tendl21(x_interpolate)

    if current_nuclide in JEFF_nuclides:
        jeff_erg, jeff_xs = General_plotter(df=JEFF, nuclides=[current_nuclide])

        f_jeff33 = scipy.interpolate.interp1d(x=jeff_erg, y=jeff_xs, fill_value='extrapolate')
        jeff_interp_xs = f_jeff33(x_interpolate)

        r2_tendl_jeff = r2_score(jeff_interp_xs, tendl_interp_xs)
        r2_endfb_jeff = r2_score(endfb8_interp_xs, jeff_interp_xs)

        print(f"TENDL21 - JEFF3.3: {r2_tendl_jeff:0.6f}")
        auxilliary_library_comparisons.append(r2_tendl_jeff)
        print(f"ENDF/B-VIII - JEFF3.3: {r2_endfb_jeff:0.6f}")
        endfb_comparisons.append(r2_endfb_jeff)


    if current_nuclide in JENDL_nuclides:
        jendl_erg, jendl_xs = General_plotter(df=JENDL, nuclides=[current_nuclide])

        f_jendl5 = scipy.interpolate.interp1d(x=jendl_erg, y=jendl_xs, fill_value='extrapolate')
        jendl_interp_xs = f_jendl5(x_interpolate)

        r2_endf_jendl = r2_score(endfb8_interp_xs, jendl_interp_xs)
        r2_tendl_jendl = r2_score(jendl_interp_xs, tendl_interp_xs)
        r2_jeff_jendl = r2_score(jeff_interp_xs, jendl_interp_xs)

        print(f"ENDF/B-VIII - JENDL5: {r2_endf_jendl:0.6f}")
        endfb_comparisons.append(r2_endf_jendl)
        if nuc in JEFF_nuclides:
            print(f"JEFF3.3 - JENDL5: {r2_jeff_jendl:0.6f}")
            auxilliary_library_comparisons.append(r2_jeff_jendl)
        print(f"TENDL21 - JENDL5: {r2_tendl_jendl:0.6f}")
        auxilliary_library_comparisons.append(r2_tendl_jendl)

    if current_nuclide in CENDL_nuclides:
        cendl_erg, cendl_xs = General_plotter(df=CENDL, nuclides=[current_nuclide])

        f_cendl32 = scipy.interpolate.interp1d(x=cendl_erg, y=cendl_xs, fill_value='extrapolate')
        cendl_interp_xs = f_cendl32(x_interpolate)

        r2_endf_cendl = r2_score(endfb8_interp_xs, cendl_interp_xs)
        endfb_comparisons.append(r2_endf_cendl)
        print(f"ENDF/B - CENDL32: {r2_endf_cendl:0.6f}")
        r2_tendl_cendl = r2_score(tendl_interp_xs, cendl_interp_xs)
        auxilliary_library_comparisons.append(r2_tendl_cendl)
        print(f"TENDL - CENDL32: {r2_tendl_cendl:0.6f}")
        if current_nuclide in JENDL_nuclides:
            r2_jendl_cendl = r2_score(jendl_interp_xs, cendl_interp_xs)
            auxilliary_library_comparisons.append(r2_jendl_cendl)
            print(f"JENDL5 - CENDL32: {r2_jendl_cendl:0.6f}")
        if current_nuclide in JEFF_nuclides:
            r2_jeff_cendl = r2_score(jeff_interp_xs, cendl_interp_xs)
            print(f"JEFF33 - CENDL32: {r2_jeff_cendl:0.6f}")
            auxilliary_library_comparisons.append(r2_jeff_cendl)

    r2_endf_tendl = r2_score(endfb8_interp_xs, tendl_interp_xs)
    endfb_comparisons.append(r2_endf_tendl)

    endfmean = np.mean(endfb_comparisons)
    endfstd = np.std(endfb_comparisons)

    auxmean = np.mean(auxilliary_library_comparisons)
    auxstd = np.std(auxilliary_library_comparisons)

    print(f"ENDF/B-VIII - TENDL21: {r2_endf_tendl:0.6f}")
    print(f"Aux mean/std: {auxmean} $\pm$ {auxstd}")
    print(f"ENDF/B mean/std: {endfmean} +- {endfstd}")

    if len(auxilliary_library_comparisons) < 2:
        if abs((endfmean - auxmean)) > 0.02:
            print(f"Outlier: {current_nuclide} - ({periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]})")
            outliers.append(current_nuclide)
    elif abs((endfmean - auxmean)) > 10 * auxstd and len(auxilliary_library_comparisons) > 1 and endfmean < auxmean and endfmean < 0.99:
        print(f"Outlier: {current_nuclide} - ({periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]})")
        outliers.append(current_nuclide)

    if endfmean > auxmean:
        false_positives.append(current_nuclide)

    print()

print(outliers)
