import pandas as pd
import periodictable
import pandas as pd
import numpy as np
import tqdm


qvaldf = pd.read_csv('MT103_Q_values.csv')

df = pd.read_csv('tendl_MT103asyms.csv')

Q_list = []

for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
	q = qvaldf[(qvaldf['Z'] == row['Z']) & (qvaldf['A'] == row['A'])]['Q'].values[0]
	Q_list.append(q)




df.insert(df.shape[1], 'Q', value=Q_list)
df.to_csv('TENDL2021_MT103_all_features.csv')