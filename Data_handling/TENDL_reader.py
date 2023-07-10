import json
import pandas as pd
import matplotlib.pyplot as plt
from matrix_functions import range_setter

with open("/Users/rntg/Project/Data_handling/TENDL19_xs.json", 'r') as file:
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

        xs_reaction = TENDL[i]["REACTION"]

        for j in range(len(xs_reaction)):
            xs_mt = xs_reaction[j]["MT"]

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

TENDL_nuclides = []
for zval, aval in zip(TENDL_z, TENDL_a):
    if [zval,aval] in TENDL_nuclides:
        continue
    else:
        TENDL_nuclides.append([zval,aval])

print(len(TENDL_nuclides))
print(TENDL_nuclides)

Nuclear_features = pd.read_csv("Nuclear_Data_features.csv")
Nuclear_features = Nuclear_features.drop(axis=1, labels=['Unnamed: 0'])

TENDL_dataframe_columns = Nuclear_features.columns

print(TENDL_dataframe_columns)

def TENDL_TO_DF():

    TENDL_dataframe = pd.DataFrame(columns= TENDL_dataframe_columns)

    pass

