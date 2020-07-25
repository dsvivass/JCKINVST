import pandas as pd

df = pd.DataFrame({'a':[1], 'b':[2]})
print(df)
df.insert(2, "b", [3], True)
# df.drop(columns='b', inplace=True)
# print(df.columns[-2], df.columns[-1])
print(df)
# df = df.loc[:,df.columns.duplicated()]
if True in df.columns.duplicated(): df.iloc[0, -2] = df.iloc[0,-1]

df = df.loc[:,~df.columns.duplicated()]
print(df)


ult_Dat = ('20200724  b', 288)
if ult_Dat[0].split('  ')[1] in df.columns:
    df[ult_Dat[0].split('  ')[1]] = ult_Dat[1]

print(ult_Dat[0].split('  ')[1])
print(df)