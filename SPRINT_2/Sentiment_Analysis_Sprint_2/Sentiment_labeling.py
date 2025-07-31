from pymongo import MongoClient, UpdateOne
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from torch.amp import autocast



# MongoDB setup
MONGO_CONNECTION_STRING = "mongodb+srv://ydandriyal:Zeus_4321@twiiter-db.qucsjdh.mongodb.net/?retryWrites=true&w=majority&appName=twiiter-db"
DB_NAME = "MYDB"
COLLECTION_NAME = "tweets"
client = MongoClient(MONGO_CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]

# Model and tokenizer
model_name = "Matteo404404/airline_tuned_roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

#batch prediction function
def batch_predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with autocast(device_type='cuda'):
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
    sentiments = [id2label[pred.item()] for pred in predictions]
    confidences = [round(conf.item(), 4) for conf in confidences]
    return sentiments, confidences


# Main loop with batching
batch_size = 256
count = 0

while True:
    batch = list(collection.find({"sentiment": {"$exists": False}}).limit(batch_size))
    if not batch:
        break

    texts = [tweet.get("text", "") for tweet in batch]
    ids = [tweet["_id"] for tweet in batch]

    sentiments, confidences = batch_predict(texts)

    updates = []
    for _id, sentiment, confidence in zip(ids, sentiments, confidences):
        updates.append(
            UpdateOne(
                {"_id": _id},
                {"$set": {"sentiment": sentiment, "confidence": confidence}}
            )
        )

    if updates:
        collection.bulk_write(updates)

    count += len(batch)
    print(f"Labeled {count} tweets...")

print("Finished labeling tweets.")
