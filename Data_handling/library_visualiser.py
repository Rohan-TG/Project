import random

import numpy as np
import pandas as pd
from matrix_functions import range_setter, General_plotter
from sklearn.metrics import r2_score
import periodictable
import scipy
import random
import matplotlib.pyplot as plt

tendl = pd.read_csv('TENDL_2021_MT16_XS_features.csv')
endfb = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")
cendl = pd.read_csv("CENDL32_all_features.csv")
jeff = pd.read_csv("JEFF33_all_features.csv")
jendl = pd.read_csv("JENDL5_arange_all_features.csv")

tendl=tendl[tendl.Z != 6]
endfb=endfb[endfb.Z != 6]
cendl=cendl[cendl.Z != 6]
jeff =jeff [jeff.Z != 6]
jendl=jendl[jendl.Z != 6]

tendl.index = range(len(tendl))
endfb.index = range(len(endfb))
cendl.index = range(len(cendl))
jeff.index = range(len(jeff))
jendl.index = range(len(jendl))

endfb_nuclides = range_setter(df=endfb, la=0, ua=260)
cendl_nuclides = range_setter(df=cendl, la=0, ua=260)
tendl_nuclides = range_setter(df=tendl, la=0, ua=260)
jeff_nuclides = range_setter(df=jeff, la=0, ua=260)
jendl_nuclides = range_setter(df=jendl, la=0, ua=260)



all_r2s = []
nuclides = []

# for x in range(20):
#     choice_nuc = random.choice(endfb_nuclides)
#     if choice_nuc not in nuclides:
#         nuclides.append(choice_nuc)
# endfb_nuclides = [[26,56]]

for i in endfb_nuclides:

    nuclide_r2_array = []

    current_nuclide = [95,242]
    print(f"{periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")

    endfb_erg, endfb_xs = General_plotter(df=endfb, nuclides=[current_nuclide])
    endfb_func = scipy.interpolate.interp1d(x=endfb_erg, y=endfb_xs, fill_value='extrapolate')


    tendlerg, tendlxs = General_plotter(df=tendl, nuclides=[current_nuclide])
    tendl_func = scipy.interpolate.interp1d(x=tendlerg, y=tendlxs, fill_value='extrapolate')

    tendl_interp_endfb = endfb_func(tendlerg)

    r2_endfb_tendl = r2_score(tendlxs[1:], tendl_interp_endfb[1:])

    # print(f"TENDL/ENDFB R2: {r2_endfb_tendl:0.5f}")

    nuclide_r2_array.append(r2_endfb_tendl)

    if current_nuclide in jendl_nuclides:
        jendlerg, jendlxs = General_plotter(df=jendl, nuclides=[current_nuclide])

        jendl_interp_endfb = endfb_func(jendlerg)
        jendl_interp_tendl = tendl_func(jendlerg)
        jendl_func = scipy.interpolate.interp1d(x=jendlerg, y=jendlxs, fill_value="extrapolate")

        r2_endfb_jendl = r2_score(jendlxs[1:], jendl_interp_endfb[1:])
        r2_tendl_jendl = r2_score(jendlxs[1:], jendl_interp_tendl[1:])
        # print(f"JENDL/ENDFB R2: {r2_endfb_jendl:0.5f}")
        # print(f"JENDL/TENDL R2: {r2_endfb_tendl:0.5f}")

        nuclide_r2_array.append(r2_endfb_jendl)
        nuclide_r2_array.append(r2_tendl_jendl)



    if current_nuclide in jeff_nuclides:
        jefferg, jeffxs = General_plotter(df=jeff, nuclides=[current_nuclide])

        jeff_func = scipy.interpolate.interp1d(x=jefferg, y=jeffxs, fill_value='extrapolate')

        jeff_interp_endfb = endfb_func(jefferg)
        jeff_interp_tendl = tendl_func(jefferg)
        jeff_interp_jendl = jendl_func(jefferg)

        r2_endfb_jeff = r2_score(jeffxs[1:], jeff_interp_endfb[1:])
        r2_tendl_jeff = r2_score(jeffxs[1:], jeff_interp_tendl[1:])
        r2_jendl_jeff = r2_score(jeffxs[1:], jeff_interp_jendl[1:])
        # print(f"JEFF/ENDFB R2: {r2_endfb_jeff:0.5f}")
        # print(f"JEFF/ENDFB R2: {r2_tendl_jeff:0.5f}")

        nuclide_r2_array.append(r2_endfb_jeff)
        nuclide_r2_array.append(r2_jendl_jeff)
        nuclide_r2_array.append(r2_tendl_jeff)


    if current_nuclide in cendl_nuclides:
        cendlerg, cendlxs = General_plotter(df=cendl, nuclides=[current_nuclide])

        cendl_interp_endfb = endfb_func(cendlerg)
        cendl_interp_tendl = tendl_func(cendlerg)
        if current_nuclide in jeff_nuclides:
            cendl_interp_jeff = jeff_func(cendlerg)
            r2_jeff_cendl = r2_score(cendlxs[1:], cendl_interp_jeff[1:])
            nuclide_r2_array.append(r2_jeff_cendl)

        cendl_interp_jendl = jendl_func(cendlerg)
        r2_endfb_cendl = r2_score(cendlxs[1:], cendl_interp_endfb[1:])
        r2_tendl_cendl = r2_score(cendlxs[1:], cendl_interp_tendl[1:])
        r2_jendl_cendl =r2_score(cendlxs[1:], cendl_interp_jendl[1:])
        # print(f"CENDL/ENDFB R2: {r2_endfb_cendl:0.5f}")
        # print(f"CENDL/TENDL R2: {r2_tendl_cendl:0.5f}")

        nuclide_r2_array.append(r2_endfb_cendl)
        nuclide_r2_array.append(r2_tendl_cendl)
        nuclide_r2_array.append(r2_jendl_cendl)

    print(np.mean(nuclide_r2_array))

    if current_nuclide in jeff_nuclides:
        plt.plot(jefferg, jeffxs, '--', label='JEFF-3.3', color='mediumvioletred')
    if current_nuclide in jendl_nuclides:
        plt.plot(jendlerg, jendlxs, label='JENDL-5', color='green')
    if current_nuclide in cendl_nuclides:
        plt.plot(cendlerg, cendlxs, '--', label='CENDL-3.2', color='gold')
    plt.plot(tendlerg, tendlxs, label="TENDL-2021", color='dimgrey', linewidth=2)
    plt.plot(endfb_erg,  endfb_xs, label = 'ENDF/B-VIII', linewidth=2)
    plt.title(f"$\sigma_{{n,2n}}$ for {periodictable.elements[current_nuclide[0]]}-{current_nuclide[1]}")
    plt.legend()
    plt.grid()
    plt.ylabel('$\sigma_{n,2n}$ / b')
    plt.xlabel('Energy / MeV')
    plt.show()
    break

    all_r2s.append(nuclide_r2_array)
    print(all_r2s)

    print()



overall_nuclide_r2 = []
for x in all_r2s:

    mean_per_nuclide = np.mean(x)
    overall_nuclide_r2.append(mean_per_nuclide)

tally = [y for y in overall_nuclide_r2 if y > 0.95]

print(f"{len(tally)}/{len(overall_nuclide_r2)}")

# 409/418 have average r2 > 0.85

# 399/418 for r2 = 0.9