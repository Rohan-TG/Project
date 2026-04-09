import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sigma_collapsed = sum(sigma_i * flux_i) / (sum(flux_i))


# Load ML XS
xs_data = pd.read_csv('Re_185_xs_values.csv')
xs_values = xs_data['XS'].values
energy_grid = xs_data['ERG'].values

fine_grid = np.arange(0, 20.01, 0.01)
interp_ml_xs = np.interp(fine_grid, energy_grid, xs_values)

# Interpolate XS values to flux energy group by integrating XS and weighting w.r.t. flux
energy_bins_data = open('ebins_709', 'r')
energy_lines = energy_bins_data.readlines()
energy_list = []
for line in energy_lines[2:]:
	linesplit = line.split(',')
	for item in linesplit:
		if not item.endswith('\n'):
			energy_list.append(float(item))
energy_list.reverse()

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

energy_list = np.array(energy_list)
energy_list = energy_list/1e6

flux_corrected_sigma = []
for ieb, flux_energy_boundary in enumerate(energy_list):
	if ieb < len(energy_list)-1:
		lower_bound = flux_energy_boundary
		upper_bound = energy_list[ieb+1]

		reduced_ml_sigma = xs_data[(xs_data['ERG'] >= lower_bound) & (xs_data['ERG'] <= upper_bound)]

		sigma_over_E = reduced_ml_sigma['XS'] / reduced_ml_sigma['ERG']

		corrected_sigma_numerator = np.trapz(sigma_over_E.values, reduced_ml_sigma['ERG'].values)

		energy_reciprocal = 1 / reduced_ml_sigma['ERG'].values
		corrected_sigma_d = np.trapz(energy_reciprocal, reduced_ml_sigma['ERG'].values)

		corrected_sigma = corrected_sigma_numerator / corrected_sigma_d

		flux_corrected_sigma.append(corrected_sigma)

fcs = np.nan_to_num(flux_corrected_sigma)

plt.figure()
plt.plot(energy_list[:-1], fcs, 'x-')
plt.grid()
plt.xlabel('Energy / eV')
plt.ylabel('(n,2n) XS')
plt.show()



widths = np.diff(energy_list)
plt.figure()
plt.bar(energy_list[:-1], fcs, width=widths)
plt.xlim(0, 25)
plt.grid()
plt.xlabel('Energy / MeV')
plt.ylabel('(n,2n) XS')
plt.title('Weighted Re-185 (n,2n) ML cross sections')
plt.show()