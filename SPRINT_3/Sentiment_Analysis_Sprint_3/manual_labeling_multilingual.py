
#clean json from label studio
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent
file_name = base_dir / "JSONs" / "label_studio_export.json"

# Load the raw Label Studio export
with open(file_name, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Extract text + label
converted_data = []
for item in raw_data:
    try:
        text = item["data"]["text"]
        label = item["annotations"][0]["result"][0]["value"]["choices"][0]
        converted_data.append({"text": text, "label": label})
    except (KeyError, IndexError):
        continue  # skip incomplete or invalid entries
file_name_2 = base_dir / "JSONs" / "tweets_for_finetuning.json"

# Save to new JSON file
with open(file_name_2, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

file_name_3 = base_dir / "CSVs" / "tweets_for_finetuning.csv"
# Save to CSV
import pandas as pd
pd.DataFrame(converted_data).to_csv(file_name_3, index=False)
