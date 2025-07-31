from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import gc
import logging
from typing import List, Dict, Optional, Tuple, Generator
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
base_dir = Path(__file__).resolve().parent.parent.parent  # Adjust to reach your project root
MONGO_URI = "mongodb+srv://mateiberari:matei2005@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
DB_NAME = "MYDB"
COLLECTION_NAME = "threads_AIR_France"
LUFTHANSA_COLLECTION_NAME = "threads_lufthansa"# this one is just threads STARTED by lufthansa, not all threads of them, this last commit have as main collection(row above) air france, but it can be used for any airline collection threads
TWEETS_COLLECTION_NAME = "tweets"
AIRLINE_ID = "106062176"
MODEL_PATH = "Matteo404404/XLM_threads_roberta_tuned"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 50

# mongo connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
lufthansa_collection = db[LUFTHANSA_COLLECTION_NAME]
tweets_collection = db[TWEETS_COLLECTION_NAME]
logger.info("MongoDB connection established")

# Loading model and tokenizer

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
logger.info(f"Model loaded successfully on {DEVICE}")


def has_sentiment_analysis(tree: Dict) -> bool:
    """Check if the tree already has sentiment analysis completed"""
    if not tree or not isinstance(tree, dict):
        return False

    try:
        # Check all paths in the tree to see if they have sentiment analysis
        for path in build_paths_generator(tree):
            for node in path:
                if not isinstance(node, dict):
                    continue

                sentiment = node.get("sentiment",{

                })

                # Check if sentiment is still "UNK"  or missing
                if not sentiment:
                    return False

                # If sentiment is a string and equals "UNK", it's not analyzed
                if isinstance(sentiment, str) and sentiment == "UNK":
                    return False

                # If sentiment is a dict, check for required analysis fields
                if isinstance(sentiment, dict):
                    required_fields = ["label_id", "score", "probabilities"]
                    if not all(field in sentiment for field in required_fields):
                        return False

        return True  # All nodes have complete sentiment analysis

    except Exception as e:
        logger.debug(f"Error checking sentiment analysis status: {e}")
        return False


def count_analyzed_documents(target_collection) -> Tuple[int, int]:
    """Count total and already analyzed documents in a collection"""
    try:
        total_docs = target_collection.count_documents({})

        # Count documents where sentiment is still "UNK"
        unanalyzed_docs = target_collection.count_documents({
            "tree.sentiment": "UNK"
        })

        analyzed_docs = total_docs - unanalyzed_docs

        return total_docs, analyzed_docs

    except Exception as e:
        logger.error(f"Error counting documents: {e}")
        return 0, 0


def setup_database_indexes():
    """Create indexes optimized"""
    try:
        logger.info("Setting up indexes for large-scale collections...")


        logger.info("Creating tweets collection indexes (this may take a few minutes for 6M documents)...")

        # bulk lookups
        tweets_collection.create_index("id", background=True, name="idx_tweets_id")


        tweets_collection.create_index("user_ref", background=True, name="idx_tweets_user_ref")

        # Optional for now, maybe for delta later for matei
        tweets_collection.create_index("created_at", background=True, name="idx_tweets_created_at")


        logger.info("Creating thread collection indexes...")
        collection.create_index([("tweet_id", 1)], background=True, name="idx_threads_tweet_id")
        lufthansa_collection.create_index([("tweet_id", 1)], background=True, name="idx_lufthansa_tweet_id")

        # Compound index for queries
        collection.create_index([
            ("tree.sentiment.label_id", 1),
            ("tweet_id", 1)
        ], sparse=True, background=True, name="idx_threads_sentiment_compound")

        lufthansa_collection.create_index([
            ("tree.sentiment.label_id", 1),
            ("tweet_id", 1)
        ], sparse=True, background=True, name="idx_lufthansa_sentiment_compound")

        # Index for checking if sentiment analysis exists
        collection.create_index([("tree.sentiment", 1)], background=True, name="idx_threads_sentiment_status")
        lufthansa_collection.create_index([("tree.sentiment", 1)], background=True,
                                          name="idx_lufthansa_sentiment_status")

        logger.info("All indexes created successfully (running in background)")
        logger.info("Note: Large collection indexes may take time to build completely")

    except Exception as e:
        logger.warning(f"Failed to create some indexes: {e}")
        logger.warning("Continuing without all indexes - performance may be degraded")


class SentimentProcessor:
    """Sentiment processor with caching"""

    def __init__(self, model, tokenizer, device, cache_size=5000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def classify_and_score(self, text: str) -> Tuple[int, float, List[float]]:
        """Classify sentiment and return label, score, and probabilities with caching"""
        if not text or not text.strip():
            return 1, 0.0, [0.33, 0.34, 0.33]  # Neutral default

        # Use first 1000 chars for caching to avoid memory issues with very long texts
        cache_text = text.strip()[:1000]
        text_hash = hash(cache_text)

        if text_hash in self.cache:
            self.cache_hits += 1
            return self.cache[text_hash]

        self.cache_misses += 1

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            label_id = probs.argmax()

            # Calculate sentiment score: -1 (negative) to +1 (positive)
            score = -1 * probs[0] + 0 * probs[1] + 1 * probs[2]

            result = (int(label_id), float(score), probs.tolist())

            # Cache management with size limit for memory efficiency
            if len(self.cache) >= self.cache_size:
                # Remove oldest 20% of entries when cache is full
                remove_count = max(1, self.cache_size // 5)
                for _ in range(remove_count):
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]

            self.cache[text_hash] = result
            return result

        except Exception as e:
            logger.error(f"Error in sentiment classification: {e}")
            return 1, 0.0, [0.33, 0.34, 0.33]  # Neutral default on error

    def get_cache_stats(self):
        """Get caching stats"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }


def get_speaker_prefix_from_node(node: Dict) -> str:
    """Get speaker prefix"""
    if not node or not isinstance(node, dict):
        return "[user]"

    user_ref = str(node.get("user_ref", ""))
    return "[airline]" if user_ref == AIRLINE_ID else "[user]"


def build_paths_generator(node: Dict, current_path: Optional[List] = None) -> Generator[List[Dict], None, None]:
    """Build all paths from root to leaves using generator"""
    if current_path is None:
        current_path = []

    if not node or not isinstance(node, dict):
        return

    current_path = current_path + [node]

    # If no children, this is a leaf so yield the complete path
    if not node.get("children") or not isinstance(node.get("children"), list):
        yield current_path
        return

    # Recursively build paths for all children
    for child in node["children"]:
        if child:
            yield from build_paths_generator(child, current_path)


def add_sentiment_fields_optimized(path: List[Dict], sentiment_processor: SentimentProcessor):
    """Add sentiment analysis results to each node in the path"""
    if not path:
        return

    concat_texts = []
    prev_score = None

    for i, node in enumerate(path):
        if not node or not isinstance(node, dict):
            continue

        try:
            # Get speaker prefix directly from node
            prefix = get_speaker_prefix_from_node(node)
            text = node.get("text", "").replace("\n", " ").strip()

            # Build cumulative conversation
            concat_texts.append(f"{prefix} {text}")
            full_input = "\n".join(concat_texts)

            # Classify sentiment
            label_id, score, probs = sentiment_processor.classify_and_score(full_input)

            # Calculate delta from previous step
            delta = None if prev_score is None else score - prev_score
            prev_score = score

            # Add sentiment fields to node
            node["sentiment"] = {
                "label_id": label_id,
                "score": round(score, 4),
                "delta": round(delta, 4) if delta is not None else None,
                "probabilities": {
                    "negative": round(probs[0], 4),
                    "neutral": round(probs[1], 4),
                    "positive": round(probs[2], 4)
                },
                "cumulative_text": full_input,
                "step_in_conversation": i,
                "cumulative_text_length": len(full_input)
            }

        except Exception as e:
            logger.error(f"Error processing node {node.get('id', 'unknown')}: {e}")
            # Add default sentiment on error
            node["sentiment"] = {
                "label_id": 1,
                "score": 0.0,
                "delta": None,
                "probabilities": {"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                "cumulative_text": "",
                "step_in_conversation": i,
                "cumulative_text_length": 0,
                "error": str(e)
            }


def process_collection_optimized(target_collection, collection_name: str,
                                 sentiment_processor: SentimentProcessor) -> Dict[str, int]:
    """Process a single collection"""
    processed_count = 0
    skipped_count = 0
    error_count = 0

    try:
        total_docs, already_analyzed = count_analyzed_documents(target_collection)
        unanalyzed_docs_count = target_collection.count_documents({"tree.sentiment": "UNK"}) # Get the count of docs to process
        logger.info(
            f"Collection {collection_name}: {total_docs} total documents, {already_analyzed} already have some sentiment analysis")
        logger.info(f"Found {unanalyzed_docs_count} documents to process.")

        if unanalyzed_docs_count == 0:
            logger.info(f"No new documents to process in {collection_name}")
            return {"processed": 0, "skipped": 0, "errors": 0}

        query_filter = {
            "tree.sentiment": "UNK"
        }


        # iterate through the documents that need processing, not all documents.
        for skip in tqdm(range(0, unanalyzed_docs_count, BATCH_SIZE), desc=f"Processing {collection_name}"):
            try:
                # Query the database for a batch in each iteration
                batch = list(target_collection.find(query_filter, {"_id": 1, "tweet_id": 1, "tree": 1}).sort('_id',1).skip(skip).limit(BATCH_SIZE))

                if not batch:
                    # This might happen if documents are updated by another process
                    logger.warning(f"Batch starting at {skip} returned no documents. May have reached the end.")
                    break

                needs_processing = []

                for doc in batch:
                    tree = doc.get("tree")
                    if not tree or not isinstance(tree, dict):
                        continue

                    if has_sentiment_analysis(tree):
                        skipped_count += 1
                        continue

                    needs_processing.append((doc, tree))

                if not needs_processing:
                    continue

                batch_updates = []
                for doc, tree in needs_processing:
                    try:
                        for path in build_paths_generator(tree):
                            add_sentiment_fields_optimized(path, sentiment_processor)

                        batch_updates.append({
                            "filter": {"_id": doc["_id"]},
                            "doc": doc
                        })
                        processed_count += 1

                    except Exception as e:
                        logger.error(f"Error processing document {doc.get('_id')}: {e}")
                        error_count += 1

                for update in batch_updates:
                    try:
                        target_collection.replace_one(
                            {"_id": update["filter"]["_id"]},
                            update["doc"]
                        )
                    except Exception as e:
                        logger.error(f"Error updating document {update['filter']['_id']}: {e}")
                        error_count += 1

                gc.collect()

                # Simplified progress logging
                logger.info(f"{collection_name}: Processed batch starting at {skip}. Total processed so far: {processed_count}")


            except Exception as e:
                logger.error(f"Error processing batch starting at {skip}: {e}")
                error_count += 1
                continue

    except Exception as e:
        logger.error(f"Error processing collection {collection_name}: {e}")
        raise

    logger.info(f"Completed {collection_name}: {processed_count} newly processed, "
                f"{skipped_count} skipped (already analyzed), {error_count} errors")

    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "errors": error_count
    }

def process_threads_and_update_optimized():
    """Main processing function"""
    setup_database_indexes()

    import time
    time.sleep(2)

    sentiment_processor = SentimentProcessor(model, tokenizer, DEVICE, cache_size=5000)

    total_stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0
    }

    try:
        # Process regular threads
        logger.info("Starting processing of user-started threads...")
        user_stats = process_collection_optimized(
            collection, "user threads", sentiment_processor
        )

        # Process Lufthansa threads
        logger.info("Starting processing of Lufthansa-started threads...")
        lufthansa_stats = process_collection_optimized(
            lufthansa_collection, "Lufthansa threads", sentiment_processor
        )

        # Combine stats
        for key in total_stats:
            total_stats[key] = user_stats[key] + lufthansa_stats[key]

        # Log cache stats
        cache_stats = sentiment_processor.get_cache_stats()
        logger.info(f"Cache statistics: {cache_stats}")

        logger.info(f"Processing complete. Total newly processed: {total_stats['processed']}, "
                    f"Total skipped: {total_stats['skipped']}, Total errors: {total_stats['errors']}")

        return total_stats

    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise


def extract_results_to_dataframe_optimized():
    """Extract sentiment results to DataFrame"""
    results = []
    collections_to_process = {
        "user_started": collection,
        "lufthansa_started": lufthansa_collection
    }

    try:
        for thread_type, target_collection in collections_to_process.items():
            logger.info(f"Extracting results from {thread_type} threads...")
            for doc in tqdm(target_collection.find({}, {"tweet_id": 1, "tree": 1}),
                            desc=f"Extracting {thread_type} results"):
                try:
                    tree = doc.get("tree")
                    if not tree:
                        continue

                    for path in build_paths_generator(tree):
                        for node in path:
                            if not isinstance(node, dict):
                                continue  # Skip if node is not a dictionary

                            sentiment = node.get("sentiment")  # Get sentiment, could be None, dict, or str


                            # Only proceed if sentiment exists AND is a dictionary
                            if sentiment and isinstance(sentiment, dict):
                                results.append({
                                    "thread_type": thread_type,
                                    "thread_id": doc.get("tweet_id"),
                                    "node_id": node.get("id"),
                                    "step": sentiment.get("step_in_conversation"),
                                    "text": node.get("text", ""),
                                    "speaker": "airline" if node.get("user_ref") == AIRLINE_ID else "user",
                                    "label_id": sentiment.get("label_id"),
                                    "score": sentiment.get("score"),
                                    "delta": sentiment.get("delta"),
                                    "neg_prob": sentiment.get("probabilities", {}).get("negative"),
                                    "neu_prob": sentiment.get("probabilities", {}).get("neutral"),
                                    "pos_prob": sentiment.get("probabilities", {}).get("positive"),
                                    "cumulative_text_length": sentiment.get("cumulative_text_length", 0),
                                    "has_error": "error" in sentiment
                                })
                except Exception as e:
                    logger.error(f"Error extracting from {thread_type} thread {doc.get('tweet_id')}: {e}")

        logger.info(f"Extraction complete. Total records: {len(results)}")
        return pd.DataFrame(results)

    except Exception as e:
        logger.error(f"Error in DataFrame extraction: {e}")
        return pd.DataFrame()

# Main
if __name__ == "__main__":
    try:
        logger.info("Starting optimized thread sentiment analysis with skip logic...")
        logger.info("SKIP LOGIC: Previously analyzed threads will be automatically skipped")
        logger.info("IMPORTANT: With 6M tweets, initial index creation may take 10-15 minutes")
        logger.info("But this will make all subsequent queries faster!")

        # Process threads and update MongoDB
        processing_stats = process_threads_and_update_optimized()

        # Extract results to DataFrame for analysis
        df_results = extract_results_to_dataframe_optimized()

        if not df_results.empty:
            # Save results to CSV
            output_file = base_dir / "CSVs"/"thread_sentiment_results_optimized.csv"
            df_results.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")

            # Print stats
            print("\n" + "=" * 50)
            print("SENTIMENT ANALYSIS RESULTS")
            print("=" * 50)
            print(f"Newly processed documents: {processing_stats['processed']}")
            print(f"Skipped documents (already analyzed): {processing_stats['skipped']}")
            print(f"Documents with errors: {processing_stats['errors']}")
            print(f"Total analyzed nodes: {len(df_results)}")
            print(f"Unique threads: {df_results['thread_id'].nunique()}")
            print(f"Records with errors: {df_results['has_error'].sum()}")

            print(f"\nSentiment distribution:")
            print(df_results['label_id'].value_counts().sort_index())

            print(f"\nAverage sentiment score: {df_results['score'].mean():.4f}")
            print(f"Score standard deviation: {df_results['score'].std():.4f}")

            print(f"\nAverage cumulative text length: {df_results['cumulative_text_length'].mean():.0f}")
            print(f"Max cumulative text length: {df_results['cumulative_text_length'].max()}")

        else:
            logger.warning("No results extracted")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise
    finally:
        try:
            client.close()
            logger.info("MongoDB connection closed")
        except:
            pass