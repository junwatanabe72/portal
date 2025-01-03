import json
from pathlib import Path

# 統合ファイルのパス
input_file_path = Path(
    "/Users/watanabeatsushi/code/python_practice/filtered/combined_tmp.json"
)
output_file_path = Path(
    "/Users/watanabeatsushi/code/python_practice/filtered/unique_combined_tmp.json"
)

# JSONファイルを読み込み
if input_file_path.exists():
    with open(input_file_path, "r", encoding="utf-8") as file:
        combined_data = json.load(file)

    # ユニークなオブジェクトを保持するセット
    unique_data = list(
        {json.dumps(obj, sort_keys=True): obj for obj in combined_data}.values()
    )

    # ユニークなオブジェクトを新しいJSONファイルに保存
    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(unique_data, file, ensure_ascii=False, indent=2)

    # オブジェクトの数をカウント
    print(f"Unique objects count: {len(unique_data)}")
    print(f"Unique data saved to: {output_file_path}")
else:
    print(f"File not found: {input_file_path}")
