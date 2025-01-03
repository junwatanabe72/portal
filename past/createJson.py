import pandas as pd

# import json
# from pandas import json_normalize

df = pd.read_json(
    "./dir/pass.json",
).T
# e = df.where(df["name"] == "東京国際クルーズターミナル")
# df_f = e.sort_values("passenger2019", ascending=False).head(10)
# print(df_f)


# 繰り返し処理
# for index, row in df.iterrows():
#     print(row)
array = [i for i in range(len(df))]
df.index = [array]
print(df.isnull().any(axis=1))
# 欠損値
# df_f = df.dropna()
# print(len(df))
# print(len(df_f))
# df.to_csv("./output.tsv", sep="\t", index=True, index_label="col0")
# print(df)
