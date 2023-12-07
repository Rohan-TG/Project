import time

import pandas as pd
import periodictable

from Data_handling import ENDF6
import os
import time
import matplotlib.pyplot as plt

ENDF6_path = r"C:\Users\TG300\Project\Data_handling\ENDFBVIII\ENDF-B-VIII.0_neutrons\ENDF-B-VIII.0_neutrons"

MT = 107
MF = 3

if os.path.exists(ENDF6_path) and os.path.isdir(ENDF6_path):
	files = os.listdir(ENDF6_path)

df = pd.DataFrame(columns=['Z', 'A', 'ERG', 'XS'])

for name in files:
	f = open(name)
	if len(name) == 15:
		element = periodictable.elements.symbol(name[6])
		nucleon_number = int(name[8:11])
		proton_number = element.number
	elif len(name) == 16:
		element = periodictable.elements.symbol(name[6:8])
		nucleon_number = int(name[9:12])
		proton_number = element.number

	print(element, nucleon_number, proton_number)

	lines = f.readlines()

	try:
		sec = ENDF6.find_section(lines, MF=MF, MT=MT)

		x, y = ENDF6.read_table(sec)

		x = [i/1e6 for i in x]

		t_ERG = []
		t_XS = []

		for ix, iy in zip(x, y):
			if ix <= 20.0:
				d = {'A': nucleon_number,
					 'Z': proton_number,
					 'XS': iy,
					 'ERG': ix}
				dummy_series = pd.Series(data=d)

				df = df._append(dummy_series, ignore_index=True)
				# print()
				# print(dummy_series)
	except:
		print(f"No MT{MT} data for {name}")




print(df)
df.to_csv('ENDFBVIII_MT107_XS_only.csv')