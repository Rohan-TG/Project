import time

import pandas as pd
import periodictable

from Data_handling import ENDF6
import os
import matplotlib.pyplot as plt

ENDF6_path = "C:/Users/TG300/Project/Data_handling/ENDF6_reading"

if os.path.exists(ENDF6_path) and os.path.isdir(ENDF6_path):
	files = os.listdir(ENDF6_path)

df = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])

for name in files:
	f = open(name)
	if len(name) == 10:
		element = periodictable.elements.symbol(name[2])
		nucleon_number = int(name[3:6])
		proton_number = element.number
	elif len(name) == 11:
		element = periodictable.elements.symbol(name[2:4])
		nucleon_number = int(name[4:7])
		proton_number = element.number

	print(element, nucleon_number, proton_number)

	lines = f.readlines()

	try:
		sec = ENDF6.find_section(lines, MF=3, MT=16)
	except:
		print(f"No MT16 data for {name}")

	x, y = ENDF6.read_table(sec)

	x = [i/1e6 for i in x]

	t_ERG = []
	t_XS = []

	for ix, iy in zip(x, y):
		if ix < 20.0:
			d = {'A': nucleon_number,
				 'Z': proton_number,
				 'XS': iy,
				 'ERG': ix}
			dummy_series = pd.Series(data=d)

			df = df._append(dummy_series, ignore_index=True)
			# print()
			# print(dummy_series)




print(df)
df.to_csv('NEW_TENDL21_A_Z_ERG_XS_only.csv')