# import pandas as pd
# import json
# from pathlib import Path
# from datetime import datetime
# import numpy as np

# # scikit-learn 系
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import classification_report

# # 不均衡データ対策（SMOTE） -- 必要な場合のみ使用
# from imblearn.over_sampling import SMOTE


# # =============================================================================
# # 1. データ読み込み関連の関数
# # =============================================================================


# # def load_training_data(paths):
# #     """
# #     指定された複数のパス(ディレクトリ)を走査し、tmp.json を見つけて読み込み、
# #     すべてのJSONデータを一つのリストにまとめて返す。
# #     """
# #     data = []
# #     for path in paths:
# #         for json_file in path.rglob("tmp.json"):  # サブディレクトリも含む
# #             with open(json_file, "r", encoding="utf-8") as file:
# #                 try:
# #                     data.extend(json.load(file))  # JSONファイル内のリスト等をextend
# #                 except Exception as e:
# #                     print(f"Error reading {json_file}: {e}")
# #     return data
# def load_training_data(paths):
#     """
#     指定された複数のパス(ディレクトリ)を走査し、tmp.json を見つけて読み込み、
#     すべてのJSONデータを一つのリストにまとめて返す。

#     ただし、'nearStation', 'prefecture', 'area', 'info', 'price' が同じデータは1つにマージする。
#     さらに、prefecture が sapporo, osaka, kyoto の物件は除外する。
#     """
#     data = []

#     # 1) まず全データを読み込む
#     for path in paths:
#         for json_file in path.rglob("tmp.json"):  # サブディレクトリも含む
#             with open(json_file, "r", encoding="utf-8") as file:
#                 try:
#                     data.extend(json.load(file))  # JSONファイル内のリスト等をextend
#                 except Exception as e:
#                     print(f"Error reading {json_file}: {e}")

#     # 2) 同じデータ (nearStation, prefecture, area, info, price が同じ) を1つにまとめる
#     merged_dict = {}

#     for obj in data:
#         # 安全に get するため、それぞれを取得 (無ければデフォルト値)
#         near_station = obj.get("nearStation", [])
#         info = obj.get("info", {})
#         price = obj.get("price", None)
#         area = obj.get("area", None)
#         prefecture = obj.get("prefecture", None)

#         # ------------------------------
#         # ★ 追加: 指定の都道府県なら除外
#         # ------------------------------
#         if prefecture in [
#             "sapporo",
#             "osaka",
#             "kyoto",
#             "osakaMetro",
#             "hyogo",
#             "fukuoka",
#             "hyogoMetro",
#             "kyotoMetro",
#             "nagoya",
#             "hiroshima",
#         ]:
#             continue  # スキップして次の物件に行く

#         # nearStation を比較用にキーへ変換
#         near_station_key = tuple(str(s) for s in near_station)

#         # info も辞書なので、(buildingCoverageRatio, floorAreaRatio) などを取り出してtupleに
#         bcr = info.get("buildingCoverageRatio", None)
#         far = info.get("floorAreaRatio", None)

#         # 一意キーを作る
#         key = (near_station_key, prefecture, area, bcr, far, price)

#         # もしキーがまだ登録されていないなら、新規登録
#         if key not in merged_dict:
#             merged_dict[key] = obj
#         else:
#             # すでに同一データがある
#             pass

#     # 3) 辞書の value 部分をリストにして返す
#     return list(merged_dict.values())


# def flatten_data(data):
#     """
#     JSONに含まれる 'info' や 'nearStation' の中身をフラットに展開して
#     1行の辞書としてまとめる。
#     """
#     flattened = []
#     for entry in data:
#         flat_entry = entry.copy()

#         if "info" in entry:
#             flat_entry.update(entry["info"])

#         if "nearStation" in entry and entry["nearStation"]:
#             station = entry["nearStation"][0]
#             flat_entry["min"] = station.get("min", None)
#             flat_entry["passenger"] = station.get("passenger", None)

#         flattened.append(flat_entry)
#     return flattened


# def load_json_by_date(base_path, date_str=None):
#     """
#     YYYY/MM/DD 形式の日付フォルダを指定して tmp.json を読み込む。
#     date_str が None の場合は今日の日付を使う。
#     """
#     if date_str is None:
#         date_str = datetime.now().strftime("%Y/%m/%d")

#     json_file_path = base_path / date_str / "tmp.json"

#     if json_file_path.exists():
#         with open(json_file_path, "r", encoding="utf-8") as file:
#             return json.load(file)
#     else:
#         print(f"File not found: {json_file_path}")
#         return []


# # =============================================================================
# # 2. メイン処理
# # =============================================================================

# # トレーニングデータを格納している複数年のフォルダパス
# training_paths = [
#     Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2023"),
#     Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2024"),
# ]

# # "今日" (あるいは任意の日付) のデータを置いているフォルダパス
# base_path_today = Path(
#     "/Users/watanabeatsushi/code/python_practice/portal/search/content"
# )

# # -----------------------------------------------------------------------------
# # 2-1. トレーニングデータの読み込みと前処理
# # -----------------------------------------------------------------------------

# # 1) トレーニングデータを読み込んでリストに集約
# training_data = load_training_data(training_paths)

# # 2) フラット化
# training_data = flatten_data(training_data)

# # 3) DataFrame 化
# df = pd.DataFrame(training_data)

# # 4) 必要なカラムが欠損していたら落とす
# required_columns = [
#     "buildingCoverageRatio",
#     "floorAreaRatio",
#     "rentPrice",
#     "area",
#     "prefecture",
#     "price",
#     "min",
#     "passenger",
#     "request",
# ]
# df = df.dropna(subset=required_columns)

# print("Before custom filtering:", df.shape)

# df = df[
#     (df["floorAreaRatio"] > 100)
#     & (df["floorAreaRatio"] < 310)
#     & (df["price"] > 10000000)
#     & (df["price"] < 160000000)
#     & (df["min"] < 6)
#     & (df["area"] > 100)
#     & (df["area"] < 300)
# ]
# print("After custom filtering:", df.shape)
# # 5) requestを二値化: request=1,2 を「1」にまとめ、それ以外(=0)を「0」
# #    これにより、 0 or 1 の2クラスに変換
# df["request_binary"] = df["request"].apply(lambda x: 1 if x in [1, 2] else 0)

# # 特徴量
# FEATURE_COLS = [
#     "buildingCoverageRatio",
#     "floorAreaRatio",
#     "rentPrice",
#     "area",
#     "prefecture",
#     "price",
#     "min",
#     "passenger",
# ]
# X = df[FEATURE_COLS]

# # ラベル(二値: 0 or 1)
# y = df["request_binary"].astype(int)
# all_prefectures = [
#     "tokyo",
#     "tokyoMetro",
#     "chiba",
#     "kanagawa",
#     "kanagawaMetro",
#     "sendai",
#     "saitama",
#     "sapporo",
#     "osaka",
#     "kyoto",
#     "osakaMetro",
#     "hyogo",
#     "fukuoka",
#     "hyogoMetro",
#     "kyotoMetro",
#     "nagoya",
#     "hiroshima",
# ]
# # 6) OneHotEncoder で 'prefecture' をエンコード
# encoder = OneHotEncoder(
#     sparse=False, handle_unknown="ignore", categories=[all_prefectures]
# )

# X_no_pref = X.drop(columns=["prefecture"]).reset_index(drop=True)
# encoded_pref = encoder.fit_transform(X[["prefecture"]])
# encoded_pref_df = pd.DataFrame(
#     encoded_pref, columns=encoder.get_feature_names_out(["prefecture"])
# ).reset_index(drop=True)

# X = pd.concat([X_no_pref, encoded_pref_df], axis=1)

# print("X shape after encoding:", X.shape)
# print("y shape:", y.shape)

# if len(X) != len(y):
#     raise ValueError("Mismatch in X and y length.")

# # 7) 学習・テスト分割
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 8) SMOTEなどの不均衡対策（必要に応じて）
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# model_X, model_y = X_train_smote, y_train_smote

# # ↑ SMOTE を使わない場合はそのまま
# # model_X, model_y = X_train, y_train


# model = LogisticRegression(
#     random_state=42,
#     max_iter=1000,
#     class_weight={0:2.0, 1:0.5}
#     # class_weight={0:1.0, 1:2.0}  # 例: クラス1の重みを小さくする
#     # class_weight="balanced",  # 不均衡対策（自動調整）
# )
# model.fit(model_X, model_y)

# # 10) テストデータ予測
# y_pred = model.predict(X_test)
# print("=== Classification Report (Binary: 0 vs 1) ===")
# print(classification_report(y_test, y_pred))

# # -----------------------------------------------------------------------------
# # 2-2. 推論: 「候補以上(=1)」を厳しく判定したい
# # -----------------------------------------------------------------------------

# today_data = load_json_by_date(base_path_today)
# today_data = flatten_data(today_data)

# # 閾値
# THRESHOLD = 0.6

# filtered_today_data = []
# for land in today_data:
#     # 必要なカラムがないなら除外
#     if not all(key in land for key in FEATURE_COLS):
#         continue

#     # land → DataFrame(1行)
#     land_features = pd.DataFrame(
#         [
#             {
#                 "buildingCoverageRatio": land["buildingCoverageRatio"],
#                 "floorAreaRatio": land["floorAreaRatio"],
#                 "rentPrice": land["rentPrice"],
#                 "area": land["area"],
#                 "prefecture": land["prefecture"],
#                 "price": land["price"],
#                 "min": land["min"],
#                 "passenger": land["passenger"],
#             }
#         ]
#     )

#     # 数値カラムを変換
#     numeric_cols = [
#         "buildingCoverageRatio",
#         "floorAreaRatio",
#         "rentPrice",
#         "area",
#         "price",
#         "min",
#         "passenger",
#     ]
#     for col in numeric_cols:
#         land_features[col] = pd.to_numeric(land_features[col], errors="coerce")

#     if land_features.isna().any().any():
#         continue

#     # prefecture エンコード
#     land_features_no_pref = land_features.drop(columns=["prefecture"]).reset_index(
#         drop=True
#     )
#     encoded_land_pref = encoder.transform(land_features[["prefecture"]])
#     encoded_land_pref_df = pd.DataFrame(
#         encoded_land_pref, columns=encoder.get_feature_names_out(["prefecture"])
#     ).reset_index(drop=True)

#     land_features = pd.concat([land_features_no_pref, encoded_land_pref_df], axis=1)

#     # 確率取得: 0 or 1
#     proba = model.predict_proba(land_features)[0]  # => [p0, p1]
#     p0, p1 = proba

#     # 「候補以上(=1)」を厳しめにする ⇒ p1 >= THRESHOLD
#     if p1 >= THRESHOLD:
#         predicted_label = 1
#     else:
#         predicted_label = 0

#     # 候補以上だけを絞り込みたい場合
#     if predicted_label == 1:
#         filtered_today_data.append(land)

# print(f"=== Filtered Lands (where p1 >= {THRESHOLD}) ===")
# # 1) title を取り出してリスト化
# titles = [land["title"] for land in filtered_today_data]

# # 2) JSON 形式でダブルクォーテーションつきの配列に変換
# #    ensure_ascii=False で日本語がそのまま表示されるようにする
# print(json.dumps(titles, ensure_ascii=False))

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# scikit-learn 系
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


# =============================================================================
# 1. データ読み込み関連の関数
# =============================================================================


def load_training_data(paths):
    """
    指定された複数のパス(ディレクトリ)を走査し、tmp.json を見つけて読み込み、
    すべてのJSONデータを一つのリストにまとめて返す。

    ただし、'nearStation', 'prefecture', 'area', 'info', 'price' が同じデータは1つにマージする。
    さらに、prefecture が sapporo, osaka, kyoto などの物件は除外する。
    """
    data = []
    for path in paths:
        for json_file in path.rglob("tmp.json"):  # サブディレクトリを含む
            with open(json_file, "r", encoding="utf-8") as file:
                try:
                    data.extend(json.load(file))
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")

    merged_dict = {}

    for obj in data:
        near_station = obj.get("nearStation", [])
        info = obj.get("info", {})
        price = obj.get("price", None)
        area = obj.get("area", None)
        prefecture = obj.get("prefecture", None)
        min = near_station[0].get("min", None)

        # 除外対象の都道府県
        # info から floorAreaRatio を取得
        far = info.get("floorAreaRatio", None)
        # if price < 15000000:
        #     continue
        # if price > 100000000:
        #     continue
        if not isinstance(min, (int, float)):
            continue
        if min > 10:
            continue

        if not isinstance(far, (int, float)):
            continue
        if far < 120:
            continue
        if far > 300:
            continue
        if area < 120:
            continue
        if area > 240:
            continue
        # 除外対象の都道府県
        if prefecture in [
            "sapporo",
            "osaka",
            "kyoto",
            "osakaMetro",
            "hyogo",
            "fukuoka",
            "hyogoMetro",
            "kyotoMetro",
            "nagoya",
            "hiroshima",
        ]:
            continue

        near_station_key = tuple(str(s) for s in near_station)
        bcr = info.get("buildingCoverageRatio", None)
        far = info.get("floorAreaRatio", None)

        key = (near_station_key, prefecture, area, bcr, far, price)

        if key not in merged_dict:
            merged_dict[key] = obj
        else:
            # 同一キーはスキップ or ここでマージ処理を書く
            pass

    return list(merged_dict.values())


def flatten_data(data):
    """
    JSONに含まれる 'info' や 'nearStation' の中身をフラットに展開して
    1行の辞書としてまとめる。
    """
    flattened = []
    for entry in data:
        flat_entry = entry.copy()

        if "info" in entry:
            flat_entry.update(entry["info"])

        if "nearStation" in entry and entry["nearStation"]:
            station = entry["nearStation"][0]
            flat_entry["min"] = station.get("min", None)
            flat_entry["passenger"] = station.get("passenger", None)

        flattened.append(flat_entry)
    return flattened


def load_json_by_date(base_path, date_str=None):
    """
    YYYY/MM/DD 形式の日付フォルダを指定して tmp.json を読み込む。
    date_str が None の場合は今日の日付を使う。
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y/%m/%d")

    json_file_path = base_path / date_str / "tmp.json"
    if json_file_path.exists():
        with open(json_file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    else:
        print(f"File not found: {json_file_path}")
        return []


# =============================================================================
# 2. ランキング上位Nを抽出する関数
# =============================================================================
def extract_top_n(model, X_test, y_test, n=10):
    """
    predict_probaからクラス1(=1)の確率が高い順に上位 n 件を抽出し、
    そのPrecisionなどを表示する（評価用）。
    """
    proba_test = model.predict_proba(X_test)  # shape: (n_samples, 2)
    p1_scores = proba_test[:, 1]

    # p1 が大きい順に並べる
    sorted_indices = np.argsort(-p1_scores)
    top_n_indices = sorted_indices[:n]

    # 上位N件の実際のラベル
    y_top_n = y_test.iloc[top_n_indices].values

    # 簡易計算: 上位N件のうち何件が実際に1だったか
    true_positives_in_topN = sum(y_top_n == 1)
    precision_topN = true_positives_in_topN / n

    print(
        f"[Ranking Top {n}] p1-score => Found {true_positives_in_topN} true positives."
    )
    print(f"Precision (top {n}) = {precision_topN:.3f}")


# =============================================================================
# 3. メイン処理
# =============================================================================

# トレーニングデータ
training_paths = [
    Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2023"),
    Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2024"),
]

base_path_today = Path(
    "/Users/watanabeatsushi/code/python_practice/portal/search/content"
)

# 1) データ読み込み
training_data = load_training_data(training_paths)
# 2) フラット化
training_data = flatten_data(training_data)

# 3) DataFrame 化
df = pd.DataFrame(training_data)

# 4) 必要カラムが欠損していたら落とす
required_columns = [
    "buildingCoverageRatio",
    "floorAreaRatio",
    "rentPrice",
    "area",
    "prefecture",
    "price",
    "min",
    "passenger",
    "request",
]
df = df.dropna(subset=required_columns)
print("Before custom filtering:", df.shape)

# 5) 任意のフィルタリング
df = df[
    (df["floorAreaRatio"] > 100)
    & (df["floorAreaRatio"] < 310)
    & (df["price"] > 10000000)
    & (df["price"] < 160000000)
    & (df["min"] < 6)
    & (df["area"] > 100)
    & (df["area"] < 201)
]
print("After custom filtering:", df.shape)

# 6) request を二値化
df["request_binary"] = df["request"].apply(lambda x: 1 if x in [1, 2] else 0)

# 特徴量とラベル
FEATURE_COLS = [
    "buildingCoverageRatio",
    "floorAreaRatio",
    "rentPrice",
    "area",
    "prefecture",
    "price",
    "min",
    "passenger",
]
X = df[FEATURE_COLS]
y = df["request_binary"].astype(int)

# prefecture の全カテゴリをリスト化 (handle_unknown="ignore" 用)
all_prefectures = [
    "tokyo",
    "tokyoMetro",
    "chiba",
    "kanagawa",
    "kanagawaMetro",
    "sendai",
    "saitama",
    "sapporo",
    "osaka",
    "kyoto",
    "osakaMetro",
    "hyogo",
    "fukuoka",
    "hyogoMetro",
    "kyotoMetro",
    "nagoya",
    "hiroshima",
]

encoder = OneHotEncoder(
    sparse=False, handle_unknown="ignore", categories=[all_prefectures]
)

X_no_pref = X.drop(columns=["prefecture"]).reset_index(drop=True)
encoded_pref = encoder.fit_transform(X[["prefecture"]])
encoded_pref_df = pd.DataFrame(
    encoded_pref, columns=encoder.get_feature_names_out(["prefecture"])
).reset_index(drop=True)

X = pd.concat([X_no_pref, encoded_pref_df], axis=1)

print("X shape after encoding:", X.shape)
print("y shape:", y.shape)
if len(X) != len(y):
    raise ValueError("Mismatch in X and y length.")

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 不均衡対策 (SMOTE)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
model_X, model_y = X_train_smote, y_train_smote

# ロジスティック回帰モデル
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight={0: 2.0, 1: 0.5},
    # class_weight="balanced"
)
model.fit(model_X, model_y)

# 評価
y_pred = model.predict(X_test)
print("=== Classification Report (Binary: 0 vs 1) ===")
print(classification_report(y_test, y_pred))

# ランキングTOP Nで精度を見る例
extract_top_n(model, X_test, y_test, n=10)

# -----------------------------------------------------------------------------
#  推論: 「閾値で抽出」 & 「JSON出力」
# -----------------------------------------------------------------------------
today_data = load_json_by_date(base_path_today)
today_data = flatten_data(today_data)

THRESHOLD = 0.6
filtered_today_data = []

for land in today_data:
    # 必要なカラムがないなら除外
    if not all(key in land for key in FEATURE_COLS):
        continue

    land_features = pd.DataFrame(
        [
            {
                "buildingCoverageRatio": land["buildingCoverageRatio"],
                "floorAreaRatio": land["floorAreaRatio"],
                "rentPrice": land["rentPrice"],
                "area": land["area"],
                "prefecture": land["prefecture"],
                "price": land["price"],
                "min": land["min"],
                "passenger": land["passenger"],
            }
        ]
    )

    numeric_cols = [
        "buildingCoverageRatio",
        "floorAreaRatio",
        "rentPrice",
        "area",
        "price",
        "min",
        "passenger",
    ]
    for col in numeric_cols:
        land_features[col] = pd.to_numeric(land_features[col], errors="coerce")
    if land_features.isna().any().any():
        continue

    # prefecture エンコード
    land_features_no_pref = land_features.drop(columns=["prefecture"]).reset_index(
        drop=True
    )
    encoded_land_pref = encoder.transform(land_features[["prefecture"]])
    encoded_land_pref_df = pd.DataFrame(
        encoded_land_pref, columns=encoder.get_feature_names_out(["prefecture"])
    ).reset_index(drop=True)

    land_features = pd.concat([land_features_no_pref, encoded_land_pref_df], axis=1)

    # 予測確率
    proba = model.predict_proba(land_features)[0]  # => [p0, p1]
    p0, p1 = proba

    # 閾値で判定
    if p1 >= THRESHOLD:
        filtered_today_data.append(land)

print(f"=== Filtered Lands (where p1 >= {THRESHOLD}) ===")
titles = [land["title"] for land in filtered_today_data]
print(json.dumps(titles, ensure_ascii=False))
