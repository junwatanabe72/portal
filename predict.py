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

# 不均衡データ対策（SMOTE） -- 必要な場合のみ使用
from imblearn.over_sampling import SMOTE


# =============================================================================
# 1. データ読み込み関連の関数
# =============================================================================


def load_training_data(paths):
    """
    指定された複数のパス(ディレクトリ)を走査し、tmp.json を見つけて読み込み、
    すべてのJSONデータを一つのリストにまとめて返す。
    """
    data = []
    for path in paths:
        for json_file in path.rglob("tmp.json"):  # サブディレクトリも含む
            with open(json_file, "r", encoding="utf-8") as file:
                try:
                    data.extend(json.load(file))  # JSONファイル内のリスト等をextend
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
    return data


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
# 2. メイン処理
# =============================================================================

# トレーニングデータを格納している複数年のフォルダパス
training_paths = [
    Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2023"),
    Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2024"),
]

# "今日" (あるいは任意の日付) のデータを置いているフォルダパス
base_path_today = Path(
    "/Users/watanabeatsushi/code/python_practice/portal/search/content/2025"
)

# -----------------------------------------------------------------------------
# 2-1. トレーニングデータの読み込みと前処理
# -----------------------------------------------------------------------------

# 1) トレーニングデータを読み込んでリストに集約
training_data = load_training_data(training_paths)

# 2) フラット化
training_data = flatten_data(training_data)

# 3) DataFrame 化
df = pd.DataFrame(training_data)

# 4) 必要なカラムが欠損していたら落とす
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

print("After dropna, df shape:", df.shape)

# 5) requestを二値化: request=1,2 を「1」にまとめ、それ以外(=0)を「0」
#    これにより、 0 or 1 の2クラスに変換
df["request_binary"] = df["request"].apply(lambda x: 1 if x in [1, 2] else 0)

# 特徴量
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

# ラベル(二値: 0 or 1)
y = df["request_binary"].astype(int)

# 6) OneHotEncoder で 'prefecture' をエンコード
encoder = OneHotEncoder(sparse=False)

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

# 7) 学習・テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8) SMOTEなどの不均衡対策（必要に応じて）
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# model_X, model_y = X_train_smote, y_train_smote

# ↑ SMOTE を使わない場合はそのまま
model_X, model_y = X_train, y_train

# 9) 分類モデル（LogisticRegression）
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced",  # 不均衡対策（自動調整）
)
model.fit(model_X, model_y)

# 10) テストデータ予測
y_pred = model.predict(X_test)
print("=== Classification Report (Binary: 0 vs 1) ===")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------------------
# 2-2. 推論: 「候補以上(=1)」を厳しく判定したい
# -----------------------------------------------------------------------------

today_data = load_json_by_date(base_path_today, date_str="01/02")
today_data = flatten_data(today_data)

# 閾値
THRESHOLD = 0.75

filtered_today_data = []
for land in today_data:
    # 必要なカラムがないなら除外
    if not all(key in land for key in FEATURE_COLS):
        continue

    # land → DataFrame(1行)
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

    # 数値カラムを変換
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

    # 確率取得: 0 or 1
    proba = model.predict_proba(land_features)[0]  # => [p0, p1]
    p0, p1 = proba

    # 「候補以上(=1)」を厳しめにする ⇒ p1 >= THRESHOLD
    if p1 >= THRESHOLD:
        predicted_label = 1
    else:
        predicted_label = 0

    # 候補以上だけを絞り込みたい場合
    if predicted_label == 1:
        filtered_today_data.append(land)

print(f"=== Filtered Lands (where p1 >= {THRESHOLD}) ===")
for land in filtered_today_data:
    print(land)
