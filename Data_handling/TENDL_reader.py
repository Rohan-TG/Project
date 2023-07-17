import json
import pandas as pd
import matplotlib.pyplot as plt
from matrix_functions import range_setter

with open("C:\\Users\\TG300\\Project\\Data_handling\\TENDL21_xs.json", 'r') as file:
    TENDL_df = json.load(file)

TENDL = TENDL_df['Nuclear_Info']["Nuclear_Data"]

# print(len(TENDL))


def TENDL_UNPACKER():

    ALL_XS = []
    ALL_ERG = []
    ALL_Z = []
    ALL_A = []

    for i in range(len(TENDL)):

        xs_z = int(TENDL[i]["Z"])
        xs_a = int(TENDL[i]["A"])
        xs_m = int(TENDL[i]["LISO"])
        # print(xs_m)

        xs_reaction = TENDL[i]["REACTION"]

        if xs_m == 0:

            for j in range(len(xs_reaction)):
                xs_mt = xs_reaction[j]["MT"]
                xs_LFS = xs_reaction[j]["LFS"]

                if xs_LFS == -1:

                    if xs_mt == 16:
                        xs_erg = xs_reaction[j]["Energy"]
                        xs_xs = xs_reaction[j]["XS"]

                        for erg, xs in zip(xs_erg, xs_xs):
                            ALL_XS.append(xs)
                            ALL_ERG.append(erg)
                            ALL_A.append(xs_a)
                            ALL_Z.append(xs_z)

    return ALL_Z, ALL_A, ALL_ERG, ALL_XS


TENDL_z,TENDL_a,TENDL_e,TENDL_xs = TENDL_UNPACKER()

TENDL_e = [ENERGY / 1000000 for ENERGY in TENDL_e]

TENDL_nuclides = []
for zval, aval in zip(TENDL_z, TENDL_a):
    if [zval,aval] in TENDL_nuclides:
        continue
    else:
        TENDL_nuclides.append([zval,aval])

TENDL_dictionary = {'Z': TENDL_z,
                    'A': TENDL_a,
                    'ERG': TENDL_e,
                    'XS': TENDL_xs}

TENDL_DATAFRAME = pd.DataFrame(data=TENDL_dictionary)

print(TENDL_DATAFRAME.head())
print(TENDL_DATAFRAME.shape)

TENDL_DATAFRAME.to_csv("TENDL21_MT16_XS_NOLISO_LFS.csv")