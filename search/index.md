tmp.json ファイルの中身を確認しました。
それによると、各物件には以下のような情報が含まれています：

title: 物件のタイトル
url: 物件のURL
image: 物件の画像URL
agent: 代理店の名前
prefecture: 都道府県または地域の名前
area: 土地面積（平方メートル）
price: 物件の価格
access: アクセス情報
nearStation: 最寄りの駅に関する情報（駅名、徒歩での所要時間、一日の乗客数など）
info: 建築基準情報（建ぺい率、容積率など）
rentPrice: 賃料

評価ロジック

m2Price = price / area
m2Rent = rentPrice/20
