import json
import time
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from pymongo import MongoClient
from itertools import islice

MONGO_URI = 'mongodb+srv://mateiberari:matei2005@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db'
DB_NAME = 'MYDB'
tweets_col_name = 'tweets'
users_col_name = 'users'
DATE_FMT = "%a %b %d %H:%M:%S %Y"

def connect_to_db(threads_col_name: str):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[tweets_col_name], db[users_col_name], db[threads_col_name]

def fetch_all_conversations(threads_col):
    return threads_col.find({}, {"tree": 1, "tweet_id": 1})

def preload_tweet_user_maps(tweets_col, users_col, thread_convos):
    print("Preloading tweets and users based on thread roots...")

    tweet_ids = [c.get("tweet_id") or c.get("tree", {}).get("id") for c in thread_convos if c.get("tree")]
    tweet_ids = [tid for tid in tweet_ids if tid is not None]

    # Only fetch relevant tweets
    tweets_cursor = tweets_col.find({"id": {"$in": tweet_ids}}, {"id": 1, "user_ref": 1})
    tweet_map = {t['id']: t.get('user_ref') for t in tweets_cursor if 'id' in t}

    # Fetch users mentioned in those tweets
    user_ids = list(set([uid for uid in tweet_map.values() if isinstance(uid, int) or isinstance(uid, str)]))

    def chunked(iterable, size):
        it = iter(iterable)
        return iter(lambda: list(islice(it, size)), [])

    user_map = {}
    for batch in chunked(user_ids, 500):
        cursor = users_col.find({"user_id": {"$in": batch}}, {"user_id": 1, "followers_count": 1})
        for u in cursor:
            uid = u.get("user_id")
            if uid is not None:
                user_map[uid] = u.get("followers_count")

    return tweet_map, user_map


def extract_user_id(node):
    user_id = node.get("user_ref")
    if isinstance(user_id, dict):
        user_id = user_id.get("user_id") or user_id.get("_id")
    return user_id or node.get("user_id") or node.get("_id")

def get_root_follower_count(convo, tweet_map, user_map):
    tweet_id = convo.get('tweet_id') or convo.get('tree', {}).get('id')
    if not tweet_id:
        return None
    user_id = tweet_map.get(tweet_id)
    return user_map.get(user_id)

def traverse_conversation(tree: Dict, airline_user_id: int, depth=0) -> Dict[str, Any]:
    stats = {
        "num_messages": 0,
        "unique_user_ids": set(),
        "num_airline_replies": 0,
        "sentiments": [],
        "airline_sentiment_indexes": [],
        "airline_reply_times": [],
        "all_timestamps": [],
        "max_depth": depth,
        "airline_response_times": [],
    }

    def helper(node, current_depth, parent_time=None):
        stats["num_messages"] += 1
        stats["max_depth"] = max(stats["max_depth"], current_depth)

        user_id = extract_user_id(node)
        stats["unique_user_ids"].add(user_id)

        sentiment = node.get("sentiment")
        stats["sentiments"].append(sentiment)

        this_time = node.get("time") or node.get("created_at")
        stats["all_timestamps"].append(this_time)

        is_airline = str(user_id) == str(airline_user_id)
        if is_airline:
            stats["num_airline_replies"] += 1
            stats["airline_sentiment_indexes"].append(len(stats["sentiments"]) - 1)
            stats["airline_reply_times"].append(this_time)
            if this_time and parent_time:
                try:
                    dt_this = datetime.strptime(this_time, DATE_FMT)
                    dt_parent = datetime.strptime(parent_time, DATE_FMT)
                    diff = (dt_this - dt_parent).total_seconds()
                    if diff >= 0:
                        stats["airline_response_times"].append(diff)
                except Exception:
                    pass

        for child in node.get("children", []):
            helper(child, current_depth + 1, this_time)

    helper(tree, depth)
    stats["root_time"] = tree.get("time") or tree.get("created_at")
    stats["root_user_id"] = extract_user_id(tree)
    stats["unique_user_ids"] = list(stats["unique_user_ids"])
    return stats

def calc_part1_metrics(convo_data: Dict, root_follower_count: int = None) -> Dict[str, Any]:
    num_messages = convo_data.get('num_messages', 0)
    num_replies = convo_data.get('num_airline_replies', 0)

    return {
        "root_follower_count": root_follower_count,
        "num_messages": num_messages,
        "num_unique_users": len(convo_data.get('unique_user_ids', [])),
        "num_airline_replies": num_replies,
        "airline_reply_rate": (num_replies / num_messages) if num_messages else 0,
        "did_airline_reply": num_replies > 0,
    }

def calc_part2_metrics(convo_data: Dict) -> Dict[str, Any]:
    def extract_score(sent):
        if sent is None:
            return None
        if isinstance(sent, dict):
            return sent.get("score") or sent.get("label_id")
        try:
            return float(sent)
        except Exception:
            return None

    sentiments = [extract_score(s) for s in convo_data.get("sentiments", [])]
    sentiments = [s for s in sentiments if s is not None]

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None

    idxs = convo_data.get("airline_sentiment_indexes", [])
    airline_sentiments = [sentiments[i] for i in idxs if i < len(sentiments)]
    avg_airline_sentiment = sum(airline_sentiments) / len(airline_sentiments) if airline_sentiments else None

    sentiment_change_after_reply = None
    if idxs:
        before = sentiments[:idxs[0]]
        after = sentiments[idxs[0] + 1:]
        if before and after:
            sentiment_change_after_reply = (sum(after) / len(after)) - (sum(before) / len(before))

    resp_times = convo_data.get("airline_response_times", [])
    avg_resp_time = (sum(resp_times) / len(resp_times)) if resp_times else None

    return {
        "avg_sentiment": avg_sentiment,
        "avg_airline_sentiment": avg_airline_sentiment,
        "sentiment_change_after_airline": sentiment_change_after_reply,
        "max_depth": convo_data.get("max_depth"),
        "avg_airline_response_time": avg_resp_time,
        "num_unique_users": len(convo_data.get("unique_user_ids", [])),
    }

def process_all_conversations(tweets_col, users_col, threads_col, airline_user_id: int, start_date=None, end_date=None):
    thread_convos = list(fetch_all_conversations(threads_col))  # Fetch once
    tweet_map, user_map = preload_tweet_user_maps(tweets_col, users_col, thread_convos)

    start_dt = datetime.strptime(start_date, DATE_FMT) if start_date else None
    end_dt = datetime.strptime(end_date, DATE_FMT) if end_date else None

    results = []
    for idx, convo in enumerate(fetch_all_conversations(threads_col), start=1):
        tree = convo.get("tree")
        if not tree:
            continue

        root_time_str = tree.get("time") or tree.get("created_at")
        if not root_time_str:
            continue
        try:
            root_time = datetime.strptime(root_time_str, DATE_FMT)
        except Exception:
            continue

        if start_dt and root_time < start_dt:
            continue
        if end_dt and root_time > end_dt:
            continue

        convo_data = traverse_conversation(tree, airline_user_id)
        root_follower_count = get_root_follower_count(convo, tweet_map, user_map)

        result = {
            "root_id": tree.get("id"),
            **calc_part1_metrics(convo_data, root_follower_count),
            **calc_part2_metrics(convo_data),
        }
        results.append(result)
        if idx % 10 == 0:
            print(f"Processed {idx} conversations...")

    return results

def save_metrics_to_csv(metrics: List[Dict], filename="Convo_metrics.csv"):
    try:
        df = pd.DataFrame(metrics)
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    start_date = args[0] if len(args) > 0 else 'Sat Jun 01 15:09:31 2019'
    end_date = args[1] if len(args) > 1 else 'Sun Jun 30 15:09:31 2019'

    start_time = time.time()

    # Toggle which airline to run
    airlines = [
        {
            "name": "lufthansa",
            "enabled": True, # Set to True to enable processing
            "user_id": 124476322,
            "threads_collection": "threads_new"
        },
        {
            "name": "airfrance",
            "enabled": False,
            "user_id": 106062176,
            "threads_collection": "threads_AIR_France"
        },
        {
            "name": "britishairways",
            "enabled": False,
            "user_id": 18332190,
            "threads_collection": "threads_british_airways"
        },
    ]


    for airline in airlines:
        if not airline["enabled"]:
            continue

        print(f"\nProcessing: {airline['name'].capitalize()} between {start_date} and {end_date}...")
        tweets_col, users_col, threads_col = connect_to_db(airline["threads_collection"])
        convo_metrics = process_all_conversations(
            tweets_col,
            users_col,
            threads_col,
            airline["user_id"],
            start_date,
            end_date
        )
        save_metrics_to_csv(convo_metrics, filename=f"metrics_{airline['name']}.csv")

    elapsed = time.time() - start_time
    print(f"\nAll finished. Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
