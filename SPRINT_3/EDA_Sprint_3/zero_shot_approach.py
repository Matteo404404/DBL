from pymongo import MongoClient
from datetime import datetime, timezone # Not strictly needed
from transformers import pipeline
import torch
import json
from pathlib import Path

# config
base_dir = Path(__file__).resolve().parent.parent.parent  # Adjust
MONGO_URI = "mongodb+srv://mateiberari:matei2005@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
DB_NAME = "MYDB"
THREADS_COLLECTION_TO_SAMPLE = "threads_new" # Or "threads_lufthansa"
SAMPLE_SIZE = 100  # How many root tweets to test
CONFIDENCE_THRESHOLD_FOR_LABEL = 0.25

# Define  candidate labels for zero-shot classification
CANDIDATE_PROBLEM_LABELS_FOR_ZERO_SHOT = [
    "flight cancellation or rebooking issue",
    "flight delay or missed connection",
    "lost or damaged luggage problem",
    "request for refund or compensation",
    "booking, ticketing, or seat issue",
    "customer service complaint",
    "in-flight experience (food, entertainment, wifi, crew)",
    "loyalty program or miles issue",
    "website or app technical problem",
    "general question or information request",
    "baggage allowance or fee query",
    "pet travel inquiry or issue",
    "positive feedback or compliment",      # Added
    "neutral observation or general mention", # Added
    "other feedback or unspecified issue"   # Modified for clarity
]

# initialize zero-shot classification pipeline
ZERO_SHOT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

print(f"Attempting to load zero-shot model: {ZERO_SHOT_MODEL_NAME}...")
device_to_use = -1 # Default to CPU
if torch.cuda.is_available():
    try:
        # Attempt to use GPU if PyTorch with CUDA is properly set up
        torch.cuda.get_device_name(0) # Check if GPU is accessible
        device_to_use = 0 # Use the first available GPU
        print(f"GPU (CUDA) is available. Using device: {torch.cuda.get_device_name(device_to_use)}")
    except Exception as e:
        print(f"Could not initialize CUDA device (Error: {e}). Falling back to CPU.")
        device_to_use = -1 # Fallback to CPU
else:
    print("GPU (CUDA) not available, using CPU for zero-shot classification.")

try:
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=ZERO_SHOT_MODEL_NAME,
        device=device_to_use
    )
    print(f"Zero-shot classification pipeline loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load zero-shot classification model: {e}")
    print("This script cannot proceed without the model.")
    zero_shot_classifier = None
    # exit() # Exit if model loading is critical for the script's purpose

# helper function to get problem label using zero-shot classification
def get_problem_label_zero_shot(tweet_text, candidate_labels, classifier_pipeline, confidence_threshold):
    if not classifier_pipeline: # Check if classifier loaded
        return "classifier_not_loaded"
    if not tweet_text or not candidate_labels:
        return "other_unknown_no_input"
    
    tweet_text = str(tweet_text).strip()
    if len(tweet_text) < 10: # Arbitrary short length, adjust if needed
        return "other_unknown_too_short"

    try:
        result = classifier_pipeline(tweet_text, candidate_labels, multi_label=False) # Get single top label

        if result and result['labels'] and result['scores']:
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            

            if top_score >= confidence_threshold:
                return top_label.lower().replace(" ", "_").replace("/", "_or_") # Make it a slug
            else:
                return f"other_unknown_low_confidence_{top_score:.2f}" # Include score if low confidence
        return "other_unknown_classification_failed"
    except Exception as e:
        # print(f"  Error during zero-shot for '{tweet_text[:50]}...': {e}")
        return "other_unknown_exception"

# test
if __name__ == "__main__":
    if not zero_shot_classifier:
        print("Exiting because zero-shot classifier failed to load.")
        exit()

    print(f"\n--- Testing Zero-Shot Problem Labeling on Root Tweets from '{THREADS_COLLECTION_TO_SAMPLE}' ---")
    print(f"--- Using confidence threshold: {CONFIDENCE_THRESHOLD_FOR_LABEL} ---")
    
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        threads_collection = db[THREADS_COLLECTION_TO_SAMPLE]
        print(f"Connected to MongoDB and collection '{THREADS_COLLECTION_TO_SAMPLE}'.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        exit()

    processed_count = 0
    labeled_results_for_review = [] # To store results for easier manual review

    start_time = datetime.now()

    for thread_doc in threads_collection.find().limit(SAMPLE_SIZE):
        processed_count += 1
        root_text = None
        root_tweet_id = None

        if "tree" in thread_doc and thread_doc["tree"]:
            root_node = thread_doc["tree"]
            root_text = root_node.get("text")
            root_tweet_id = root_node.get("id") # Twitter ID from the tree node




            if root_text:
                predicted_label = get_problem_label_zero_shot(
                    root_text,
                    CANDIDATE_PROBLEM_LABELS_FOR_ZERO_SHOT,
                    zero_shot_classifier,
                    CONFIDENCE_THRESHOLD_FOR_LABEL
                )
                labeled_results_for_review.append({
                    "root_tweet_id": root_tweet_id,
                    "text": root_text,
                    "predicted_label": predicted_label
                })
                # Print for immediate feedback during testing
                if processed_count <= 20 or processed_count % 10 == 0 : # Print first 20, then every 10th
                    print(f"\nTweet ID: {root_tweet_id}")
                    print(f"Text: {root_text[:200]}...")
                    print(f"Predicted Label: {predicted_label}")
            else:
                print(f"Skipping tweet ID {root_tweet_id} (from doc _id {thread_doc.get('_id')}) - no text found in root node.")
        else:
            print(f"Skipping document _id {thread_doc.get('_id')} - no tree structure found.")
        
        if processed_count % 50 == 0 and processed_count > 0:
            print(f"--- Processed {processed_count}/{SAMPLE_SIZE} tweets ---")
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            if elapsed_seconds > 0:
                tweets_per_second = processed_count / elapsed_seconds
                print(f"    Speed: {tweets_per_second:.2f} tweets/second")


    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    print(f"\n--- Finished Zero-Shot Labeling Test for {processed_count} tweets in {total_duration:.2f} seconds ---")
    if processed_count > 0 and total_duration > 0:
        print(f"--- Average speed: {processed_count / total_duration:.2f} tweets/second ---")
    
    # Save results for easier review
    output_filename = base_dir / "JSONs"/"zero_shot_labeling_test_results.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(labeled_results_for_review, f, indent=2)
        print(f"Test results for review saved to {output_filename}")
    except Exception as e:
        print(f"Error saving test results: {e}")

    if client:
        client.close()
        print("MongoDB connection closed.")