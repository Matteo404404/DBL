import os
import json
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://ydandriyal:Zeus_4321@twiiter-db.qucsjdh.mongodb.net/?retryWrites=true&w=majority&appName=twiiter-db"
DB_NAME = "MYDB"
COLLECTION_NAME = "tweets"
INPUT_DIR = "Data/Clean_Data"

# CONNECTION
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

file_count = 0
total_inserted = 0

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json") or filename.endswith(".jsonl"):
        file_path = os.path.join(INPUT_DIR, filename)
        file_count += 1
        print(f"Processing {filename}...")

        docs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if docs:
            try:
                result = collection.insert_many(docs, ordered=False)
                total_inserted += len(result.inserted_ids)
                print(f"Inserted {len(result.inserted_ids)} tweets from {filename}")
            except Exception as e:
                print(f"Error during bulk insert: {e}")

print(f"Total: {total_inserted} tweets inserted from {file_count} files.")
