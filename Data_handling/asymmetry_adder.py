import pandas as pd


df = pd.read_csv("TENDL21_arange_zeroed_with_LDPs.csv")

print(df.shape)


nucs = []

for i, j, in zip(df['A'], df['Z']):  # i is A, j is Z
    if [j, i] in nucs:
        continue
    else:
        nucs.append([j, i])  # format is [Z, A]

def asymmetry_term(N, Z, A):

    s_target = (N - Z)/A

    compound_N = N+1
    compound_A = A+1
    s_compound = (compound_N - Z) / compound_A

    daughter_N = N-1
    daughter_A = A-1
    s_daughter = (daughter_N - Z)/daughter_A

    return s_target, s_compound, s_daughter

s, s_c, s_d = asymmetry_term(N=df['N'], Z=df['Z'], A=df['A'])

df.insert(67, 'Asymmetry', value=s)
df.insert(68, 'Asymmetry_compound', value=s_c)
df.insert(69, 'Asymmetry_daughter', value=s_d)

df.index = range(len(df))
df = df.drop(columns=['Unnamed: 0'])
print(df.columns)
print(df.shape)

df.to_csv("TENDL21_MT16_XS_features_zeroed.csv")