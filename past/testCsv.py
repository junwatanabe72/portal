# import math
import pandas as pd

# import matplotlib.pyplot as plt

# import japanize_matplotlib


# import json
# from pandas import json_normalize
path = "./dir/13_Tokyo_20213_20214.csv"
df = pd.read_csv(path)
df_f = df[df["種類"] == "宅地(土地)"]
df_pl = df_f[df_f["前面道路：種類"] != "私道"]
df_pv = df_f[df_f["前面道路：種類"] == "私道"]
# df_f = df_f["取引価格（㎡単価）"].astype("int")
df_f.to_csv("./output.tsv", sep="\t", index=True, index_label="col0")
# df_f = df.sort_values("passenger2019", ascending=False).head(10)
# df_f[["passenger2019"]].plot.bar()  # 棒グラフ
# plt.show()
# ddd = df_f.apply(lambda x: math.ceil(x["取引価格（㎡単価）"]), axis=1)
print(df_f["取引価格（㎡単価）"].mean())
print(df_pl["取引価格（㎡単価）"].mean())
print(df_pv["取引価格（㎡単価）"].mean())
# df_ff["passenger2017"].T.plot.barh()  # 行と列を総入れ替えして横棒グラフ
# plt.show()
# print(df_f)
