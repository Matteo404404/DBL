from pymongo import MongoClient

# === CONFIG ===
MONGO_URI    = "mongodb+srv://ydandriyal:Zeus_4321@twiiter-db.qucsjdh.mongodb.net/?retryWrites=true&w=majority&appName=twiiter-db"
DB_NAME      = "MYDB"
THREADS_COLL = "threads"
BATCH_SIZE   = 500

client      = MongoClient(MONGO_URI)
db          = client[DB_NAME]
threads_col = db[THREADS_COLL]

def count_descendants(node):
    """Recursively count all descendants in the tree node."""
    if not node.get("children"):
        return 0
    total = len(node["children"])
    for child in node["children"]:
        total += count_descendants(child)
    return total

def update_total_descendants(last_id=None):
    """
    Iterates threads collection, computes total descendants per tree,
    updates documents with 'total_descendants' field.
    If last_id is given, starts after that _id (for incremental updates).
    """
    query = {}
    if last_id:
        query["_id"] = {"$gt": last_id}

    cursor = threads_col.find(query).sort("_id", 1).batch_size(BATCH_SIZE)
    count = 0
    last_processed_id = None

    for thread_doc in cursor:
        tree = thread_doc.get("tree")
        if not tree:
            continue

        total_desc = count_descendants(tree)
        threads_col.update_one(
            {"_id": thread_doc["_id"]},
            {"$set": {"total_descendants": total_desc}}
        )
        count += 1
        last_processed_id = thread_doc["_id"]

        if count % 100 == 0:
            print(f"✅ Processed {count} threads, last _id: {last_processed_id}")

    print(f"🎉 Updated total descendants for {count} threads.")
    return last_processed_id

if __name__ == "__main__":
    update_total_descendants()
