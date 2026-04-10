import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sigma_collapsed = sum(sigma_i * flux_i) / (sum(flux_i))


# Load ML XS
xs_filename = 'Pr-141_predictions.csv'
xs_data = pd.read_csv(xs_filename)
xs_values = xs_data['XS'].values
energy_grid = xs_data['ERG'].values



fine_grid = np.arange(0, 20.01, 0.01)
interp_ml_xs = np.interp(fine_grid, energy_grid, xs_values)
xs_data_interp = pd.DataFrame({'XS': interp_ml_xs, 'ERG': fine_grid})


# Interpolate XS values to flux energy group by integrating XS and weighting w.r.t. flux

def fetch_ebins(filename):
	energy_bins_data = open(f'{filename}', 'r')
	energy_lines = energy_bins_data.readlines()
	energy_list = []
	for line in energy_lines[2:]:
		linesplit = line.split(',')
		for item in linesplit:
			if not item.endswith('\n'):
				energy_list.append(float(item))
	energy_list.reverse()

	return energy_list

energy_list = fetch_ebins('ebins_709')

def fetch_fluxes(flux_filename):
	fluxes = open(f'{flux_filename}', 'r')
	flux_lines = fluxes.readlines()
	flux_list = []
	for line in flux_lines[:-2]:
		split_line = line.split()
		for item in split_line:
			flux_list.append(float(item))
	flux_list.reverse()

	return flux_list

flux_list = fetch_fluxes('2000exp_5min_fluxes')


widths = np.diff(energy_list)

# Flux plot
plt.figure()
plt.bar(energy_list[:-1], flux_list, width=widths)
plt.grid()
plt.xlabel('Energy / eV')
plt.ylabel('Unnormalised spectrum')
plt.xscale('log')
plt.yscale('log')
plt.show()

energy_list = np.array(energy_list)
energy_list = energy_list/1e6

def correct_sigma(ml_dataframe):

	flux_corrected_sigma = []
	for ieb, flux_energy_boundary in enumerate(energy_list):
		if ieb < len(energy_list)-1:
			lower_bound = flux_energy_boundary
			upper_bound = energy_list[ieb+1]

			reduced_ml_sigma = ml_dataframe[(ml_dataframe['ERG'] >= lower_bound) & (ml_dataframe['ERG'] <= upper_bound)]

			sigma_over_E = reduced_ml_sigma['XS'] / reduced_ml_sigma['ERG']

			corrected_sigma_numerator = np.trapz(sigma_over_E.values, reduced_ml_sigma['ERG'].values)

			energy_reciprocal = 1 / reduced_ml_sigma['ERG'].values
			corrected_sigma_d = np.trapz(energy_reciprocal, reduced_ml_sigma['ERG'].values)

			corrected_sigma = corrected_sigma_numerator / corrected_sigma_d

			flux_corrected_sigma.append(corrected_sigma)

	fcs = np.nan_to_num(flux_corrected_sigma)
	return fcs

fcs = correct_sigma(ml_dataframe=xs_data_interp)

plt.figure()
plt.plot(energy_list[:-1], fcs, 'x-')
plt.grid()
plt.xlabel('Energy / MeV')
plt.ylabel('(n,2n) XS')
plt.title('Interpolated correction')
plt.xlim(0, 25)
plt.show()



widths = np.diff(energy_list)
plt.figure()
plt.bar(energy_list[:-1], fcs, width=widths)
plt.xlim(0, 25)
plt.grid()
plt.xlabel('Energy / MeV')
plt.ylabel('(n,2n) XS')
plt.title('Weighted Pr-141 (n,2n) ML cross sections')
plt.show()



def collapse(flux_values, corrected_xs):

	numer = np.sum(np.array(corrected_xs) * np.array(flux_values))
	denom = np.sum(flux_values)

	return numer/denom

collapsed_sigma = collapse(flux_list, fcs)

# tendl_value = 2.070711
#
# isomer_fraction = 0.18210701541644395
# normal_fraction = 0.8178929845835561

norm_flux = np.array(flux_list) / max(flux_list)
norm_xs = np.array(fcs) / max(fcs)



# fig, ax1 = plt.subplots()
#
# ax1.set_xlabel('Energy')
# ax1.set_ylabel('Norm flux', color='red')
# ax1.plot(energy_list[:-1], norm_xs, color='red')
# ax1.tick_params(axis='y', labelcolor='red')
#
# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('n,2n', color=color)  # we already handled the x-label with ax1
# ax2.plot(energy_list[:-1], norm_xs, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.xlim(0, 25)
# plt.show()
#
#
#
# plt.figure()
# plt.bar(energy_list[:-1], norm_flux, width=widths)
# plt.grid()
# plt.xlabel('Energy / MeV')
# plt.xlim(0, 25)
# plt.yscale('log')
# plt.show()