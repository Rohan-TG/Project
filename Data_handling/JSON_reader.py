import json
import pandas as pd
import matplotlib.pyplot as plt
from matrix_functions import range_setter

with open("/Users/rntg/Project/Data_handling/JENDL4.json", 'r') as file:
    ENDF_df = json.load(file)

JSON = ENDF_df['Nuclear_Info']["Nuclear_Data"]



def JSON_UNPACKER():

    ALL_XS = []
    ALL_ERG = []
    ALL_Z = []
    ALL_A = []

    for i in range(len(JSON)):

        xs_z = int(JSON[i]["Z"])
        xs_a = int(JSON[i]["A"])
        xs_m = int(JSON[i]["LISO"])
        # print(xs_m)

        xs_reaction = JSON[i]["REACTION"]

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


_z,_a,_e,_xs = JSON_UNPACKER()

_e = [ENERGY / 1000000 for ENERGY in _e]

JSON_nuclides = []
for zval, aval in zip(_z, _a):
    if [zval,aval] in JSON_nuclides:
        continue
    else:
        JSON_nuclides.append([zval,aval])

ENDF_dictionary = {'Z': _z,
                    'A': _a,
                    'ERG': _e,
                    'XS': _xs}

ENDF_DATAFRAME = pd.DataFrame(data=ENDF_dictionary)

print(ENDF_DATAFRAME.head())
print(ENDF_DATAFRAME.shape)

# ENDF_DATAFRAME.to_csv("JENDL4.csv")
# print("complete")