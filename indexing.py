from pymongo import MongoClient, ASCENDING

# === CONFIG ===
MONGO_URI   = "mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
DB_NAME     = "MYDB"

client      = MongoClient(MONGO_URI)
db          = client[DB_NAME]

# === USERS COLLECTION ===
# Unique lookup by user_id
# db.users.create_index(
#     [("user_id", ASCENDING)],
#     unique=True,
#     name="user_id_1"
# )

# === ENTITIES COLLECTION ===
# Unique dedupe key

# print("Dropping old 'entity_key_1' index if it exists…")
# try:
#     db.entities.drop_index("entity_key_1")
#     print(" → dropped.")
# except Exception:
#     print(" → no such index to drop.")

# print("Creating new partial index on entities._key where _key is a string…")
# db.entities.create_index(
#     [("_key", ASCENDING)],
#     name="entity_key_nonnull",
#     unique=True,
#     partialFilterExpression={ "_key": { "$type": "string" } }
# )


# === TWEETS COLLECTION ===
# 1) Unique tweet lookup by tweet_id
# db.tweets.create_index(
#     [("tweet_id", ASCENDING)],
#     unique=True,
#     name="tweet_id_1"
# )

# 2) Fast root-query & conversation mining
db.tweets.create_index(
    [("tweet_data.in_reply_to_status_id", ASCENDING), ("created_at", ASCENDING)],
    name="reply_to_createdAt_idx"
)

# 3) Lookup by entities_ref (so you can filter roots with entities quickly)
db.tweets.create_index(
    [("entities_ref", ASCENDING)],
    name="entities_ref_idx"
)

# 4) Lookup by user_ref (if you need to query all tweets by a user)
db.tweets.create_index(
    [("user_ref", ASCENDING)],
    name="user_ref_idx"
)

print("✅ All indexes created or already existed.")
