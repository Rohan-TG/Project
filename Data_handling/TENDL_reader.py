import json
import pandas as pd
import matplotlib.pyplot as plt

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

        if i > 0:
            print("Done")
            break

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




z,a,e,xs = TENDL_UNPACKER()

print(len(xs))
print(len(z))

plt.plot(e[:60], xs[:60])
plt.show()