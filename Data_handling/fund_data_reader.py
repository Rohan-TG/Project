import pandas as pd

# df = pd.read_csv("1_fund.csv")

# print(df.columns)


# al = []
# for i, j, in zip(df['A'], df['Z']): # i is A, j is Z
# 	if [j, i] in al:
# 		continue
# 	else:
# 		al.append([j, i]) # format is [Z, A]
#
#
# print(al)


df = pd.read_json("TENDL19_xs.json")

print(df.columns)

df_head = df['Nuclear_Info']

print(df_head['Nuclear_Data'])
