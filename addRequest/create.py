import json
from pathlib import Path

# 基本パス（ルートディレクトリ）
base_paths = [
    Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2023"),
    Path("/Users/watanabeatsushi/code/python_practice/portal/search/content/2024"),
]

# 処理対象の日付のリストを生成 (1年分、ゼロ埋め形式)
dates = [f"{month:02}/{day:02}" for month in range(1, 13) for day in range(1, 32)]

# 出力先のファイル
output_file_path = Path(
    "/Users/watanabeatsushi/code/python_practice/filtered/combined_tmp.json"
)
output_file_path.parent.mkdir(parents=True, exist_ok=True)

# 全データを格納するリスト
combined_data = []

# ファイル処理
for base_path in base_paths:
    for date in dates:
        try:
            json_file_path = base_path / date / "tmp.json"
            print(f"Checking file: {json_file_path}")  # デバッグ: ファイル確認
            if json_file_path.exists():  # ファイルが存在する場合のみ処理
                with open(json_file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                # `request: 1` のオブジェクトを抽出して追加
                filtered_data = [obj for obj in data if obj.get("request") == 1]
                combined_data.extend(filtered_data)

        except Exception as e:
            print(f"Error processing {json_file_path}: {e}")

# まとめたデータを新しいJSONファイルに保存
with open(output_file_path, "w", encoding="utf-8") as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=2)

print(f"Combined data saved to: {output_file_path}")
