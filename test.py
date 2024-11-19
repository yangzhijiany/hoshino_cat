import json

data_path = "E:\\hoshino_cat_project\\fine_tuning_data.jsonl"

dataset = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()  # 去掉空格或换行符
        if not line:  # 忽略空行
            continue
        try:
            dataset.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping invalid line: {line} - Error: {e}")

print("Loaded examples:", len(dataset))
print("First example:", dataset[0])