import pandas as pd
import tqdm


qvaldf = pd.read_csv('/Users/rntg/PycharmProjects/Project/MT107_regression/data/MT107_Q_values.csv')

df = pd.read_csv('/Users/rntg/PycharmProjects/Project/MT107_regression/data/JEFF-3.3_correctedxs_MT107.csv')

Q_list = []

for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
	q = qvaldf[(qvaldf['Z'] == row['Z']) & (qvaldf['A'] == row['A'])]['Q'].values[0]
	Q_list.append(q)




df.insert(df.shape[1], 'Q', value=Q_list)
df.to_csv('JEFF-3.3_MT107_all_features.csv')