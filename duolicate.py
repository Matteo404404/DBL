# from pymongo import MongoClient

# MONGO_URI = "mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
# client = MongoClient(MONGO_URI)

# db = client["MYDB"]
# collection = db["tweets"]

# pipeline = [
#     {
#         "$group": {
#             "_id": "$id",
#             "count": {"$sum": 1},
#             "docs": {"$push": "$_id"}
#         }
#     },
#     {
#         "$match": {
#             "count": {"$gt": 1}
#         }
#     }
# ]

# duplicates = list(collection.aggregate(pipeline))

# print(f"Found {len(duplicates)} duplicate 'id' values.")

# for dup in duplicates:
#     print(f"id: {dup['_id']} appears {dup['count']} times, doc _ids: {dup['docs']}")

from pymongo import MongoClient

client     = MongoClient("mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db")
db         = client["MYDB"]
tweets_col = db["tweets"]

# 1) Create a non‐unique ascending index on "id"
tweets_col.create_index([("id", 1)], name="id_1")

print("✅ Created index on tweets.id")
