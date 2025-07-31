from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb+srv://ydandriyal:Zeus_4321@twiiter-db.qucsjdh.mongodb.net/?retryWrites=true&w=majority&appName=twiiter-db")
db = client["MYDB"]



# Collections
tweets_collection = db["tweets"]
users_collection = db["users"]
entities_collection = db["entities"]
media_collection = db["media"]

# Loop through each tweet in the tweets collection
for tweet in tweets_collection.find():
    #Insert user data into the 'users' collection if not already present
    user_data = tweet.get("user", {})
    user_id = user_data.get("user_id")

    if user_id and not users_collection.find_one({"user_id": user_id}):
        user_doc = {
            "user_id": user_id,
            "name": user_data.get("name"),
            "screen_name": user_data.get("screen_name"),
            "protected": user_data.get("protected"),
            "verified": user_data.get("verified"),
            "followers_count": user_data.get("followers_count"),
            "friends_count": user_data.get("friends_count"),
            "place": user_data.get("place"),
        }
        user_doc_id = users_collection.insert_one(user_doc).inserted_id
    else:
        user_doc_id = users_collection.find_one({"user_id": user_id})["_id"]

    #Insert entities data into the 'entities' collection if not already present
    entities_data = tweet.get("entities", {})
    if entities_data:
        hashtags = entities_data.get("hashtags", [])
        urls = entities_data.get("urls", [])
        user_mentions = entities_data.get("user_mentions", [])

        entities_doc = {
            "hashtags": hashtags,
            "urls": urls,
            "user_mentions": user_mentions
        }
        entities_doc_id = entities_collection.insert_one(entities_doc).inserted_id
    else:
        entities_doc_id = None

    # Insert media data into the 'media' collection if media is present
    media_data = tweet.get("media", {})
    if media_data:
        media_doc = {
            "media_type": media_data.get("media_type"),
            "media_count": media_data.get("media_count")
        }
        media_doc_id = media_collection.insert_one(media_doc).inserted_id
    else:
        media_doc_id = None

    # Insert the tweet data into the 'tweets' collection with references to other collections
    tweet_doc = {
        "created_at": tweet.get("created_at"),
        "id": tweet.get("id"),
        "text": tweet.get("text"),
        "tweet_data": tweet.get("tweet_data", {}),
        "user_id": user_doc_id,  # Reference to the 'users' collection
        "entities_id": entities_doc_id,  # Reference to the 'entities' collection
        "media_id": media_doc_id  # Reference to the 'media' collection
    }

    # Update
    tweets_collection.update_one(
        {"_id": tweet["_id"]},
        {"$set": tweet_doc}
    )

print("Data has been separated and referenced in new collections.")
