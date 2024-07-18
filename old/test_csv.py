import csv

# ヘッダ付csvをDictとして読み込む
with open("class_mapping.csv", "r") as f:
    csvreader = csv.DictReader(f)
    rows = list(csvreader)
    print(rows)
    print(type(rows[0]))

new_dict={}
for item in rows:
    name = item["class_id"]
    new_dict[name] = item.pop("answer")

print(new_dict)

print(new_dict.get("0"))