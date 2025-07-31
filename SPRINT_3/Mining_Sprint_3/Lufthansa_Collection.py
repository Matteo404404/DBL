
# Build threads_lufthansa collection: trees rooted at Lufthansa's own tweets


from pymongo import MongoClient

# config
MONGO_URI         = "mongodb+srv://hanhminh:aJqHcovAkAksxSKZ@twiiter-db.qucsjdh.mongodb.net/?retryWrites=true&w=majority&appName=twiiter-db"
DB_NAME           = "MYDB"
TWEETS_COLL       = "tweets"
USERS_COLL        = "users"
THREADS_LH_COLL   = "threads_lufthansa"
ROOT_BATCH        = 1000

# Connect
client            = MongoClient(MONGO_URI)
db                = client[DB_NAME]
tweets_col        = db[TWEETS_COLL]
users_col         = db[USERS_COLL]
threads_lh_col    = db[THREADS_LH_COLL]


# 1) Recursive Tree Builder

def build_tree(tweet_id):
    doc = tweets_col.find_one(
        {"id": tweet_id},
        {"id": 1, "text": 1, "created_at": 1, "sentiment": 1}
    )
    if not doc:
        return None

    node = {
        "id":        doc["id"],
        "text":      (doc.get("text", "") or "")[:100],
        "time":      doc.get("created_at"),
        "sentiment": doc.get("sentiment", "UNK"),
        "children":  []
    }

    for child in tweets_col.find(
        {"tweet_data.in_reply_to_status_id": tweet_id},
        sort=[("created_at", 1)]
    ).batch_size(500):
        child_id = child.get("id")
        subtree = build_tree(child_id)
        if subtree:
            node["children"].append(subtree)

    return node


# 2) Hardcoded Lufthansa Main Account ID


LUFT_USER_ID = 124476322

print(f"Using hardcoded Lufthansa user ID: {LUFT_USER_ID}")


# 3) Fetch Lufthansa-authored root tweets (not replies)


roots_cursor = tweets_col.find(
    {
        "tweet_data.in_reply_to_status_id": None,
        "user_ref": LUFT_USER_ID  # ← Now using direct match
    },
    {"id": 1}
).batch_size(ROOT_BATCH)

count = 0
for root in roots_cursor:
    tweet_id = root["id"]
    tree = build_tree(tweet_id)
    if tree:
        threads_lh_col.replace_one(
            {"_id": tweet_id},
            {"_id": tweet_id, "tree": tree},
            upsert=True
        )
        count += 1
        if count % ROOT_BATCH == 0:
            print(f"Processed {count} Lufthansa roots…")

print(f"Stored {count} Lufthansa-authored conversation trees in “{THREADS_LH_COLL}” collection.")