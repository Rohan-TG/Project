import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sigma_collapsed = sum(sigma_i * flux_i) / (sum(flux_i))


# Load ML XS
xs_data = pd.read_csv('Re_185_xs_values.csv')


# Interpolate XS values to flux energy group by integrating XS and weighting w.r.t. flux


energy_bins_data = open('ebins_709', 'r')
energy_lines = energy_bins_data.readlines()
energy_list = []
for line in energy_lines[2:]:
	linesplit = line.split(',')
	for item in linesplit:
		if not item.endswith('\n'):
			energy_list.append(float(item))


fluxes = open('1996exp_7hour_fluxes', 'r')
flux_lines = fluxes.readlines()

flux_list = []
for line in flux_lines[:-2]:
	split_line = line.split()
	for item in split_line:
		flux_list.append(float(item))


widths = np.diff(energy_list)

plt.figure()
plt.bar(energy_list[:-1], flux_list, width=widths)
plt.grid()
plt.xlabel('Energy / eV')
plt.ylabel('Normalised spectrum')
plt.xscale('log')
plt.yscale('log')
plt.show()
