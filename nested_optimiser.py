import pandas as pd
import numpy as np
import random
import periodictable
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from matrix_functions import anomaly_remover, OPT_make_train, OPT_make_test


df = pd.read_csv("ENDFBVIII_zeroed_LDP_XS.csv")
df_test = pd.read_csv("ENDFBVIII_zeroed_LDP_XS.csv")

al = []
for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
	if [j, i] in al or i > 210:
		continue
	else:
		al.append([j, i]) # format is [Z, A]



space = {'n_estimators': hp.choice('n_estimators', [500, 600, 650, 700, 750, 800, 850, 900, 1000, 1100, 1200, ]),
		 'subsample': hp.loguniform('subsample', np.log(0.05), np.log(1.0)),
		 'max_leaves': 0,
		 'max_depth': scope.int(hp.quniform("max_depth", 6, 12, 1)),
		 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.03))}


def optimiser(space):

	validation_nuclides = []
	validation_set_size = 1000000000000

	test_nuclides = [[26, 56], [78, 195], [24, 52], [92, 235], [92, 238],
					 [40, 90], ]

	nuclides_used = []

	while len(validation_nuclides) < validation_set_size:
		choice = random.choice(al)  # randomly select nuclide from list of all nuclides
		if choice not in validation_nuclides and choice not in nuclides_used:
			validation_nuclides.append(choice)
			nuclides_used.append(choice)
	print("Nuclide selection complete")