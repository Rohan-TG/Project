import pandas as pd
from matrix_functions import range_setter

tendl = pd.read_csv('TENDL21_MT16_XS_features_zeroed.csv')
endfb = pd.read_csv("ENDFBVIII_MT16_XS_feateng.csv")

endfb_nuclides = range_setter(df=endfb, la=30, ua=210)

tendl = tendl[tendl.Z == 45]

tendl = tendl[tendl.A == 93]

print(tendl)


nuc = [45, 93]

match_element = []
for nuclide in endfb_nuclides:
    if nuclide[0] == nuc[0]:
        match_element.append(nuclide)
min_difference = 100
for match_nuclide in match_element:
    difference = match_nuclide[1] - nuc[1]
    if abs(difference) < abs(min_difference):
        min_difference = difference
        min_difference_nuclide = match_nuclide

print(match_element)
print(min_difference_nuclide)