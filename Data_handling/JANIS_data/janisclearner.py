import pandas as pd
import tqdm

from matrix_functions import range_setter
import numpy as np



part1 = pd.read_csv('Part1MT16_JANISdata_dataframe.csv')
part2 = pd.read_csv('MT16JANIS_part2_dataframe.csv')
# error_part = pd.read_csv('MT16_error_data.csv')



def orderer(df):
	nuclides = range_setter(df=df, la=0, ua= 290)

	Z = df['Z']
	A = df['A']
	Energy = df['ERG']
	XS = df['XS']
	dXS = df['dXS']

	allz = []
	alla = []
	alldxs = []
	allerg = []
	allxs = []

	for nuc in tqdm.tqdm(nuclides, total=len(nuclides)):
		for j, (z, a) in enumerate(zip(Z,A)):
			if z == nuc[0] and a == nuc[1] and Energy[j] <= 20:
				allz.append(Z[j])
				alla.append(A[j])
				alldxs.append(dXS[j])
				allxs.append(XS[j])
				allerg.append(Energy[j])

	tempdf = pd.DataFrame({'Z': allz,
						   'A': alla,
						   'ERG': allerg,
						   'XS': allxs,
						   'dXS': alldxs,})

	return tempdf