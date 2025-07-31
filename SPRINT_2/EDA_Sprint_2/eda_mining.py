from pymongo import MongoClient

# Config
MONGO_URI    = "mongodb+srv://ydandriyal:Zeus_4321@twiiter-db.qucsjdh.mongodb.net/?retryWrites=true&w=majority&appName=twiiter-db"
DB_NAME = "MYDB"
THREADS_COLL = "threads"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
threads_col = db[THREADS_COLL]

# Total conversations
total_threads = threads_col.count_documents({})
print(f"Total conversation threads: {total_threads}")

# Conversations with 10+ replies
threads_10plus = threads_col.count_documents({
    "tree.children": {"$exists": True, "$not": {"$size": 0}},
    "$expr": {"$gte": [{"$size": "$tree.children"}, 3]}
})
print(f"Conversations with 10+ direct replies: {threads_10plus}")
threads_10plus_pct = threads_10plus / total_threads * 100 if total_threads else 0
print(f"Empty conversation trees: {threads_10plus} ({threads_10plus_pct:.2f}%)")

# % of empty trees (no children)
empty_trees_count = threads_col.count_documents({
    "$or": [
        {"tree.children": {"$exists": False}},
        {"tree.children": {"$size": 0}}
    ]
})
empty_trees_pct = empty_trees_count / total_threads * 100 if total_threads else 0
print(f"Empty conversation trees: {empty_trees_count} ({empty_trees_pct:.2f}%)")

roots_lufthansa_count = threads_col.count_documents({
    "tree.user.screen_name": {"$regex": "^lufthansa$", "$options": "i"}
})
print(f"Conversation roots posted by Lufthansa: {roots_lufthansa_count}")
