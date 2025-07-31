import pymongo
import pandas as pd
from tqdm import tqdm
import html
import ftfy
import re
import unicodedata
from pathlib import Path

# config
base_dir = Path(__file__).resolve().parent.parent.parent  # Adjust
MONGO_URI = "mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
DB_NAME = "MYDB"
THREADS_COLLECTION_NAME = "threads_new"
TWEETS_COLLECTION_NAME = "tweets"
AIRLINE_ID = "124476322"
OUTPUT_CSV_FILE = base_dir / "CSVs"/"random_threads_cumulative_cleaned_for_manual_labeling.csv"
NUMBER_OF_THREADS_TO_FETCH = 100


def enhanced_text_cleaning(text):
    """
    Simplified and more robust text cleaning
    """
    if not text or not text.strip():
        return ""

    #Fix encoding issues and mojibake
    cleaned_text = ftfy.fix_text(text)

    #Unescape HTML entities
    cleaned_text = html.unescape(cleaned_text)

    #Normalize Unicode
    cleaned_text = unicodedata.normalize('NFC', cleaned_text)

    #Replace user mentions with a special token

    cleaned_text = re.sub(r'@\w+', '[USER]', cleaned_text)

    #Replace URLs with a special token
    cleaned_text = re.sub(r'http[s]?://\S+', '[URL]', cleaned_text)

    #Normalize all whitespace to a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()


    return cleaned_text


def get_speaker_prefix(tweet_id, tweets_collection_obj, airline_id_str):
    if not tweet_id:
        return "[user]"
    tweet_doc = tweets_collection_obj.find_one({"id": str(tweet_id)})
    if tweet_doc:
        user_ref = tweet_doc.get("user_ref")
        if user_ref is not None and str(user_ref) == airline_id_str:
            return "[airline]"
        else:
            return "[user]"
    return "[user]"


def build_paths_from_tree_node(node, current_path_nodes=None):
    if current_path_nodes is None:
        current_path_nodes = []
    if node is None or 'id' not in node or 'text' not in node:
        if not current_path_nodes: return []
        return [current_path_nodes]
    current_path_nodes = current_path_nodes + [node]
    children = node.get("children")
    if not children:
        return [current_path_nodes]
    all_paths_of_nodes = []
    for child_node in children:
        all_paths_of_nodes.extend(build_paths_from_tree_node(child_node, list(current_path_nodes)))
    if not all_paths_of_nodes and current_path_nodes:
        return [current_path_nodes]
    return all_paths_of_nodes


def main():
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        threads_collection = db[THREADS_COLLECTION_NAME]
        tweets_collection = db[TWEETS_COLLECTION_NAME]
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return

    print(f"Fetching {NUMBER_OF_THREADS_TO_FETCH} random threads...")
    try:
        if threads_collection.count_documents({}) == 0:
            print(f"The collection '{THREADS_COLLECTION_NAME}' is empty.")
            client.close()
            return
        random_threads_cursor = threads_collection.aggregate([
            {"$sample": {"size": NUMBER_OF_THREADS_TO_FETCH}}
        ])
        random_threads = list(random_threads_cursor)
        if not random_threads:
            print("Could not fetch random threads.")
            client.close()
            return
        print(f"Fetched {len(random_threads)} threads.")
    except Exception as e:
        print(f"Error fetching random threads: {e}")
        client.close()
        return

    processed_data_for_csv = []
    skipped_entries = 0

    print("Processing fetched threads with enhanced cleaning...")
    for thread_doc in tqdm(random_threads, desc="Processing Threads"):
        conversation_tree = thread_doc.get("tree")
        if not conversation_tree:
            continue
        all_conversation_paths_nodes = build_paths_from_tree_node(conversation_tree)
        if not all_conversation_paths_nodes:
            continue

        for path_nodes in all_conversation_paths_nodes:
            current_cumulative_path_texts = []
            for node in path_nodes:
                tweet_id = node.get("id")
                original_text = node.get("text", "")

                if not original_text.strip():
                    continue

                # Use enhanced cleaning function
                cleaned_text = enhanced_text_cleaning(original_text)

                if not cleaned_text or len(cleaned_text) < 2:  # More lenient threshold
                    skipped_entries += 1
                    continue

                speaker_prefix = get_speaker_prefix(tweet_id, tweets_collection, AIRLINE_ID)
                single_prefixed_turn = f"{speaker_prefix} {cleaned_text}"

                current_cumulative_path_texts.append(single_prefixed_turn)
                cumulative_text_this_step = "\n".join(current_cumulative_path_texts)

                processed_data_for_csv.append({
                    "text": cumulative_text_this_step,
                    "label": ""
                })

    if not processed_data_for_csv:
        print("No data was processed. The output CSV will be empty or not generated.")
        client.close()
        return

    df = pd.DataFrame(processed_data_for_csv, columns=['text', 'label'])

    # Additional filtering optimized for roberta fine-tuning
    original_count = len(df)

    # Remove very short texts
    df = df[df['text'].str.len() >= 10]

    # Remove very long texts that would be truncated anyway just to be sure
    df = df[df['text'].str.len() <= 1800]

    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])

    # Remove texts with too many special characters (likely corrupted)
    def is_mostly_text(text):
        alpha_count = sum(c.isalpha() or c.isspace() for c in text)
        return alpha_count / len(text) > 0.3  # At least 30% letters/spaces

    df = df[df['text'].apply(is_mostly_text)]

    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"\nSuccessfully processed and cleaned {len(df)} cumulative text entries.")
        print(f"Skipped {skipped_entries} entries due to cleaning issues.")
        print(f"Removed {original_count - len(df)} entries due to length/quality/duplicate filtering.")
        print(f"Final dataset size optimized for XLM-RoBERTa: {len(df)} entries")
        print(f"Average text length: {df['text'].str.len().mean():.1f} characters")
        print(f"Output saved to: {OUTPUT_CSV_FILE}")

        # Show sample of cleaned data
        print("\nSample of cleaned data:")
        for i, text in enumerate(df.head(3)['text']):
            print(f"Sample {i + 1} ({len(text)} chars): {text[:100]}...")
            print("---")

    except Exception as e:
        print(f"Error saving data to CSV: {e}")

    client.close()
    print("MongoDB connection closed.")


if __name__ == "__main__":
    main()