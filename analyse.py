# すべてのjsonを結合するスクリプト
import pandas as pd
import glob
import json

# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

files = glob.glob("./search/content/2023/*/*/tmp.json")
# データフレームを格納するための空リストを作成
data = []
num = 0
prefectures = {
    "tokyoMetro": "東京23区",
    "tokyo": "東京都市部",
    "saitama": "埼玉県",
    "chiba": "千葉県",
    "kanagawaMetro": "横浜市内",
    "kanagawa": "神奈川県",
    "osakaMetro": "大阪市内",
    "osaka": "大阪市外",
    "hyogoMetro": "神戸市",
    "hyogo": "神戸市外",
    "kyoto": "京都区外",
    "kyotoMetro": "京都区内",
    "sapporo": "札幌市",
    "sendai": "仙台市",
    "hiroshima": "広島市",
    "nagoya": "名古屋市",
    "fukuoka": "福岡市",
}
# 各ファイルに対してループ
for file in files:
    with open(file, "r") as f:
        # jsonファイルを読み込み
        json_data = json.load(f)
        num = num + 1
        # データの構造を確認
        # print(json_data)
        # 各jsonオブジェクトに対してループ
        for obj in json_data:
            # 'price', 'rentPrice', 'prefecture'のみを取得
            data.append(
                {key: obj[key] for key in ["price", "rentPrice", "area", "prefecture"]}
            )

# データフレームを作成
df = pd.DataFrame(data)
df["price_per_area"] = (df["price"] / 10000) / (df["area"] / 3.305785)
df["rentPrice_per_area"] = df["rentPrice"] / df["area"]
# 通常の浮動小数点数として表示するように設定
pd.options.display.float_format = "{:.0f}".format


# 'prefecture'列の値を辞書に基づいて変換
df["prefecture"] = df["prefecture"].map(prefectures)

# 'prefecture'でグループ化し、'price'と'rentPrice'の平均を計算
averages = df.groupby("prefecture")[["price_per_area", "rentPrice_per_area"]].mean()
ordered_prefectures = [prefectures[key] for key in prefectures.keys()]

#
averages = averages.reindex(ordered_prefectures)
counts = df.groupby("prefecture").size()
counts = counts.reindex(ordered_prefectures)
#

# 結果を表示
print(counts)
# 結果を表示
print(averages)
print(num)
# x = []
# y = []
# target_col = ["area", "price", "agent", "title", "prefecture"]
# # fig = plt.figure()
# plt.bar(y, x)
# plt.xticks(rotation=25)
# plt.ylabel("平均土地単価(万円/坪)")
# plt.xlabel("各エリア")
# plt.title("投資対象エリア土地新着物件")  # 売上推移
# # plt.text(1, 100, "test", horizontalalignment="center", color="black")
# [
#     plt.text(i, 20, str(value), horizontalalignment="center", color="black")
#     for i, value in enumerate(x)
# ]
# # plt.show()
# plt.savefig("./img.png")
