#!/usr/bin/env python3
from pymongo import MongoClient, UpdateOne
from datetime import datetime

# config
MONGO_URI     = "mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority"
DB_NAME       = "MYDB"
THREADS_COLL  = "threads_new"      # or threads_new
BATCH_SIZE    = 1000

# Lufthansa’s ObjectId or numeric user_ref
LUFTHANSA_ID  = 124476322

# set up
client       = MongoClient(MONGO_URI)
db           = client[DB_NAME]
threads_col  = db[THREADS_COLL]

def parse_created_at(dt):
    if isinstance(dt, datetime):
        return dt
    # strip timezone if present
    clean = dt.replace("+0000 ","")
    return datetime.strptime(clean, "%a %b %d %H:%M:%S %Y")

def backfill_from_tree():
    bulk_ops = []
    n_updated = 0

    # only threads not yet back-filled:
    cursor = threads_col.find(
        {"response_delay_h": {"$exists": False}},
        {"_id":1, "tweet_id":1, "tree.time":1, "tree.children":1}
    ).batch_size(BATCH_SIZE)

    for thread in cursor:
        root_oid   = thread["_id"]
        root_time  = parse_created_at(thread["tree"]["time"])
        children   = thread["tree"].get("children", [])

        # find first Lufthansa reply among direct children (already sorted by time)
        first = None
        for child in children:
            if child.get("user_ref") == LUFTHANSA_ID:
                first = child
                break

        if not first:
            # no airline reply in direct children → skip
            continue

        reply_time = parse_created_at(first["time"])
        delay_h    = (reply_time - root_time).total_seconds() / 3600.0

        op = UpdateOne(
            {"_id": root_oid},
            {"$set": {
                "first_reply_id":        first["_id"],
                "first_reply_sentiment": first.get("sentiment","UNK"),
                "response_delay_h":      round(delay_h, 3)
            }}
        )
        bulk_ops.append(op)
        n_updated += 1

        if len(bulk_ops) >= BATCH_SIZE:
            threads_col.bulk_write(bulk_ops, ordered=False)
            print(f"Updated {n_updated} threads so far…")
            bulk_ops.clear()

    # flush remainder
    if bulk_ops:
        threads_col.bulk_write(bulk_ops, ordered=False)

    print(f"Finished back-filling. Total threads updated: {n_updated}")

if __name__ == "__main__":
    backfill_from_tree()
