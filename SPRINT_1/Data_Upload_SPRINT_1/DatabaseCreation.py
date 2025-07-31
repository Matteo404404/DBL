import argparse
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# config
MONGO_URI  = "mongodb+srv://ydandriyal:Zeus_4321@twiiter-db.qucsjdh.mongodb.net/?retryWrites=true&w=majority&appName=twiiter-db"
DB_NAME    = "MYDB"
BATCH_SIZE = 1000

client      = MongoClient(MONGO_URI)
db          = client[DB_NAME]
tweets_col  = db["tweets"]
users_col   = db["users"]
entities_col= db["entities"]
media_col   = db["media"]

def revert_references():
    """Remove any user_ref, entities_ref, media_ref"""
    print("Reverting previous normalization…")
    res = tweets_col.update_many(
        {"$or":[
            {"user_ref":{"$exists":True}},
            {"entities_ref":{"$exists":True}},
            {"media_ref":{"$exists":True}}
        ]},
        {"$unset":{"user_ref":"","entities_ref":"","media_ref":""}}
    )
    print(f"Reverted on {res.modified_count} tweets.")

def upsert_user(user):
    """Upsert user doc by user_id, return its _id."""
    uid = user.get("user_id")
    if not uid: return None
    doc = {
        "_id": uid,
        "user_id": uid,
        "name": user.get("name"),
        "screen_name": user.get("screen_name"),
        "protected": user.get("protected"),
        "verified": user.get("verified"),
        "followers_count": user.get("followers_count"),
        "friends_count": user.get("friends_count"),
        "place": user.get("place"),
        "created_at": user.get("created_at")
    }
    users_col.replace_one({"_id":uid}, doc, upsert=True)
    return uid

def insert_entities(ent):
    """Insert entities doc, return its ObjectId, or None."""
    if not ent or not (ent.get("hashtags") or ent.get("urls") or ent.get("user_mentions")):
        return None
    return entities_col.insert_one({
        "hashtags": ent.get("hashtags",[]),
        "urls":     ent.get("urls",[]),
        "user_mentions": ent.get("user_mentions",[])
    }).inserted_id

def insert_media(md):
    """Insert media doc, return its ObjectId, or None."""
    if not md: return None
    return media_col.insert_one({
        "media_type":  md.get("media_type"),
        "media_count": md.get("media_count")
    }).inserted_id

def normalize_collections():
    """Page through tweets by _id, extract refs, and bulk-update in batches."""
    print("Starting normalization…")
    last_id = None
    total   = 0

    while True:
        query = {}
        if last_id is not None:
            query["_id"] = {"$gt": last_id}

        batch = list(
            tweets_col
            .find(query)
            .sort("_id", 1)
            .limit(BATCH_SIZE)
        )
        if not batch:
            break

        ops = []
        for tw in batch:
            upd = {}
            # extract user
            u = tw.get("user")
            if u: upd["user_ref"]     = upsert_user(u)
            # extract entities
            e = tw.get("entities")
            if e: upd["entities_ref"] = insert_entities(e)
            # extract media
            m = tw.get("media")
            if m: upd["media_ref"]    = insert_media(m)
            # remove embedded to reclaim space
            unset = {k:"" for k in ("user","entities","media") if tw.get(k) is not None}
            setp  = {k:v for k,v in upd.items() if v is not None}

            if setp or unset:
                ops.append(
                    UpdateOne({"_id": tw["_id"]},
                              {"$set": setp, "$unset": unset})
                )

        if ops:
            try:
                res = tweets_col.bulk_write(ops, ordered=False)
                total += res.modified_count
            except BulkWriteError as bwe:
                print("Bulk write error:", bwe.details)

        last_id = batch[-1]["_id"]
        print(f"Processed {total} tweets so far…")

    print(f"Normalization complete. Total tweets updated: {total}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--revert",    action="store_true", help="Remove old refs")
    p.add_argument("--normalize", action="store_true", help="Extract and set refs")
    args = p.parse_args()

    if args.revert:    revert_references()
    if args.normalize: normalize_collections()
    if not (args.revert or args.normalize):
        p.print_help()
