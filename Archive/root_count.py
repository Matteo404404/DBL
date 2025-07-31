from pymongo import MongoClient

MONGO_URI    = "mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "MYDB"
COLLECTION_NAME = "tweets"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
tweets_col = db[COLLECTION_NAME]

root_count = tweets_col.count_documents({
    "text": {"$regex": "@AirFrance", "$options": "i"},
    "tweet_data.in_reply_to_status_id": None,
})

print(f"Total root tweets: {root_count}")







