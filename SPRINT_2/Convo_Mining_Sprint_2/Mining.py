# #!/usr/bin/env python3
from pymongo import MongoClient, UpdateOne
import networkx as nx
import matplotlib.pyplot as plt 

MONGO_URI      = "mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
DB_NAME        = "MYDB"
TWEETS_COLL    = "tweets"
ENTITIES_COLL  = "entities"
THREADS_COLL   = "threads_new"
BATCH_SIZE     = 1000

# Lufthansa’s user_id on Twitter:
AIRLINE_ID   = 124476322

client         = MongoClient(MONGO_URI)
db             = client[DB_NAME]
tweets_col     = db[TWEETS_COLL]
entities_col   = db[ENTITIES_COLL]
threads_col    = db[THREADS_COLL]

def build_tree(root_id, visited=None):
    """Recursively build a conversation tree, including user_ref & entities_ref."""
    if visited is None:
        visited = set()
    if root_id in visited:
        return None
    visited.add(root_id)

    # only pull the fields we need
    doc = tweets_col.find_one(
        {"id": root_id},
        {"_id":1, "id":1, "text":1, "created_at":1,
         "sentiment":1, "user_ref":1, "entities_ref":1}
    )
    if not doc:
        return None

    node = {
        "_id":         doc["_id"],
        "id":          doc["id"],
        "text":        doc.get("text","")[:100],
        "time":        doc.get("created_at"),
        "sentiment":   doc.get("sentiment","UNK"),
        "user_ref":    doc.get("user_ref"),
        "entities_ref":doc.get("entities_ref"),
        "children":    []
    }

    # fetch direct replies (by matching tweet_data.in_reply_to_status_id)
    for reply in tweets_col.find(
        {"tweet_data.in_reply_to_status_id": root_id},
        {"id":1}
    ).sort("created_at", 1).batch_size(1000):
        child = build_tree(reply["id"], visited)
        if child:
            node["children"].append(child)

    return node

def process_roots_in_batches():
    print(f"Looking for BT_Airways‐mention roots in batches of {BATCH_SIZE}…")
    last_id       = None
    count_stored  = 0
    count_no_ent  = 0

    while True:
        # Base query: root‐level tweets with a non‐null entities_ref
        qry = {
            "tweet_data.in_reply_to_status_id": None,
            "entities_ref": {"$ne": None}
        }
        if last_id:
            qry["id"] = {"$gt": last_id}

        batch = list(
            tweets_col.find(qry, {"_id":1, "id":1, "entities_ref":1})
                      .sort("id", 1)
                      .limit(BATCH_SIZE)
        )
        if not batch:
            break

        for root in batch:
            # Dereference entities_ref and check if Lufthansa is mentioned
            ent_doc = entities_col.find_one(
                {"_id": root["entities_ref"], "user_mentions.id": AIRLINE_ID},
                {"_id":1}
            )
            if not ent_doc:
                # this root has no valid entities or no Lufthansa mention
                count_no_ent += 1
            else:
                # build the full tree and upsert into threads
                tree = build_tree(root["id"])
                if tree:
                    threads_col.replace_one(
                        {"_id": root["_id"]},
                        {"_id": root["_id"], "tweet_id": root["id"], "tree": tree},
                        upsert=True
                    )
                    count_stored += 1

        last_id = batch[-1]["id"]
        print(f"  • Scanned up to id={last_id}: stored {count_stored}, skipped (no entities) {count_no_ent}")

    print(f"\nDone. Total trees stored: {count_stored}")
    print(f"Roots lacking entities or bt_Airways mention: {count_no_ent}")

if __name__ == "__main__":
    process_roots_in_batches()
