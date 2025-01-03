import pandas as pd

path = "./dir/13_Tokyo_20213_20214.csv"
df = pd.read_csv(path, encoding="utf-8")
df_f = df[df["種類"] == "宅地(土地)"]
df_f.to_json("./output.json", orient="records", force_ascii=False)
# print(df_f)
