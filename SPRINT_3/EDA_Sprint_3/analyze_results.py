import json
import os
from collections import Counter
import re # For basic text cleaning
import pandas as pd # For easier aggregation and comparison
from pathlib import Path

# config
base_dir = Path(__file__).resolve().parent.parent.parent
BASE_RESULTS_DIR = base_dir / "CSVs"/ "multi_airline_analysis_results_final" # this might change



\
# Ensure 'output_dir_suffix' matches what was used for saving.
AIRLINE_CONFIGS_FOR_ANALYSIS = {
    "Lufthansa": {
        "output_dir_suffix": "lufthansa",
        "name_for_stopwords": "lufthansa" # For cleaning text specific to this airline
    },
    "BritishAirways": {
        "output_dir_suffix": "british_airways",
        "name_for_stopwords": "british airways ba" # Add multiple aliases
    },
    "KLM": {
        "output_dir_suffix": "klm",
        "name_for_stopwords": "klm"
    },
    "AmericanAir": {
        "output_dir_suffix": "american_air",
        "name_for_stopwords": "american air aa"
    }
    # Add other airlines here if they were processed
}

# helper
def load_json_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return []
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

# tokenization and cleaning function
def clean_and_tokenize(text, airline_specific_stopwords=""):
    base_stopwords = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "ve", "ll",
        "d", "m", "o", "re", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
        "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
    ])

    dynamic_stopwords = base_stopwords.union(set(airline_specific_stopwords.lower().split()))
    
    text = str(text).lower() # Ensure text is string
    text = re.sub(r"(@\w+|#\w+|http\S+)", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in dynamic_stopwords and len(word) > 2]
    return tokens

# ANALYSIS FUNCTION

def analyze_ineffective_replies_data(data, airline_name, airline_stopwords):
    if not data: return
    print(f"\n--- {airline_name} Analysis: Ineffective Airline Replies (Total: {len(data)}) ---")
    print("\nSample of Ineffective Reply Sequences:")
    for i, seq in enumerate(data[:min(3, len(data))]): # Show up to 3 samples
        print(f"  Sequence {i+1} (Thread: {seq.get('thread_root_tweet_id')}):")
        print(f"    User1 ({seq.get('initial_user_tweet_id')}): '{seq.get('initial_user_text')}' (Sent: {seq.get('initial_user_sentiment')})")
        print(f"    {airline_name} Reply ({seq.get('airline_reply_id')}): '{seq.get('airline_reply_text')}' (Sent: {seq.get('airline_reply_sentiment')}, RespTime: {seq.get('airline_response_time_sec')}s)")
        print(f"    User2 ({seq.get('final_user_tweet_id')}): '{seq.get('final_user_text')}' (Sent: {seq.get('final_user_sentiment_score')}, CIS: {seq.get('final_user_cis')}) Reason: {seq.get('reason')}")

    initial_texts = [seq.get('initial_user_text', '') for seq in data]
    all_tokens = []
    for text in initial_texts: all_tokens.extend(clean_and_tokenize(text, airline_stopwords))
    word_counts = Counter(all_tokens)
    print(f"\nMost common words in Initial User Complaints for {airline_name} (ineffective replies):")
    for word, count in word_counts.most_common(5): print(f"  '{word}': {count}")
    return {"airline": airline_name, "type": "ineffective_replies", "count": len(data), "top_complaint_words": dict(word_counts.most_common(5))}


def analyze_unanswered_tweets_data(data, airline_name, airline_stopwords):
    if not data: return
    print(f"\n--- {airline_name} Analysis: Unanswered Negative User Tweets (Total: {len(data)}) ---")
    print("\nSample of Unanswered Negative Tweets:")
    for i, tweet in enumerate(data[:min(3, len(data))]):
        print(f"  Tweet {i+1} (Thread: {tweet.get('thread_root_tweet_id')}): ID: {tweet.get('tweet_id')}, User: {tweet.get('user_id')}, Sent: {tweet.get('sentiment_score')}, Text: '{tweet.get('text')}'")

    texts = [tweet.get('text', '') for tweet in data]
    all_tokens = []
    for text in texts: all_tokens.extend(clean_and_tokenize(text, airline_stopwords))
    word_counts = Counter(all_tokens)
    print(f"\nMost common words in Unanswered Negative Tweets for {airline_name}:")
    for word, count in word_counts.most_common(5): print(f"  '{word}': {count}")
    return {"airline": airline_name, "type": "unanswered_negative", "count": len(data), "top_words": dict(word_counts.most_common(5))}


def analyze_escalations_data(data, airline_name, airline_stopwords):
    if not data: return
    print(f"\n--- {airline_name} Analysis: Negative Sentiment Escalations (Total: {len(data)}) ---")
    print("\nSample of Negative Escalations:")
    for i, esc in enumerate(data[:min(3, len(data))]):
        print(f"  Escalation {i+1} (Thread: {esc.get('thread_root_tweet_id')}):")
        print(f"    Escalating ({esc.get('escalating_tweet_id')}, Author: {esc.get('escalating_tweet_author_type')}): '{esc.get('escalating_tweet_text')}' (Sent: {esc.get('escalating_tweet_sentiment')}, CIS: {esc.get('cis_at_escalation')})")
        print(f"    Preceded by ({esc.get('parent_tweet_id')}, Author: {esc.get('parent_tweet_author_type')}): '{esc.get('parent_tweet_text')}' (Sent: {esc.get('parent_tweet_sentiment')})")

    parent_texts = [esc.get('parent_tweet_text', '') for esc in data]
    all_tokens = []
    for text in parent_texts: all_tokens.extend(clean_and_tokenize(text, airline_stopwords))
    word_counts = Counter(all_tokens)
    print(f"\nMost common words in Tweets PRECEDING an Escalation for {airline_name}:")
    for word, count in word_counts.most_common(5): print(f"  '{word}': {count}")
    return {"airline": airline_name, "type": "escalations", "count": len(data), "top_preceding_words": dict(word_counts.most_common(5))}


# main exe
if __name__ == "__main__":
    print("--- Starting Multi-Airline Analysis of Pattern Detection Results ---")
    
    all_airline_summary_stats = []

    for airline_name_key, config in AIRLINE_CONFIGS_FOR_ANALYSIS.items():
        print(f"\n\n Analyzing data for {airline_name_key} ".center(70, "=")) # Analyzing data
        airline_output_dir = os.path.join(BASE_RESULTS_DIR, config['output_dir_suffix'])
        airline_stopword_list = config.get('name_for_stopwords', airline_name_key) # Use specific stopwords

        if not os.path.exists(airline_output_dir):
            print(f"Warning: Output directory not found for {airline_name_key}: {airline_output_dir}. Skipping.")
            all_airline_summary_stats.append({"airline": airline_name_key, "error": "Output directory not found"})
            continue

        ineffective_file = os.path.join(base_dir, "JSONs", airline_output_dir, "ineffective_replies.json")
        unanswered_file = os.path.join(base_dir, "JSONs", airline_output_dir, "unanswered_negative_tweets.json")
        escalations_file = os.path.join(base_dir,"JSONs", airline_output_dir, "negative_escalations.json")

        ineffective_data = load_json_data(ineffective_file)
        unanswered_data = load_json_data(unanswered_file)
        escalations_data = load_json_data(escalations_file)
        
        airline_stats = {"airline": airline_name_key, "patterns": []}

        if ineffective_data:
            summary = analyze_ineffective_replies_data(ineffective_data, airline_name_key, airline_stopword_list)
            if summary: airline_stats["patterns"].append(summary)
        
        if unanswered_data:
            summary = analyze_unanswered_tweets_data(unanswered_data, airline_name_key, airline_stopword_list)
            if summary: airline_stats["patterns"].append(summary)

        if escalations_data:
            summary = analyze_escalations_data(escalations_data, airline_name_key, airline_stopword_list)
            if summary: airline_stats["patterns"].append(summary)
            
        all_airline_summary_stats.append(airline_stats)

    # Comparative Summary
    print(f"\n\n Analyzing data for {airline_name_key} ".center(70, "="))
    print("\n\n" + " Comparative Summary by Airline ".center(70, "=")) # Comparative Summary by Airline
    
    # Create a DataFrame for easier comparison
    summary_for_df = []
    for airline_data in all_airline_summary_stats:
        row = {"Airline": airline_data.get("airline")}
        if "error" in airline_data:
            row["Ineffective Replies Count"] = "N/A (Error)"
            row["Unanswered Negative Count"] = "N/A (Error)"
            row["Escalations Count"] = "N/A (Error)"
        else:
            for pattern_summary in airline_data.get("patterns", []):
                if pattern_summary.get("type") == "ineffective_replies":
                    row["Ineffective Replies Count"] = pattern_summary.get("count", 0)
                elif pattern_summary.get("type") == "unanswered_negative":
                    row["Unanswered Negative Count"] = pattern_summary.get("count", 0)
                elif pattern_summary.get("type") == "escalations":
                    row["Escalations Count"] = pattern_summary.get("count", 0)
        summary_for_df.append(row)
    
    df_summary = pd.DataFrame(summary_for_df)
    print(df_summary.to_string()) # Print DataFrame as string

    #save this DataFrame to a CSV
    try:
        summary_csv_path = os.path.abspath(os.path.join(base_dir, "CSVs", BASE_RESULTS_DIR, "comparative_summary_stats.csv"))
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\nComparative summary saved to: {summary_csv_path}")
    except Exception as e:
        print(f"Error saving comparative summary CSV: {e}")

    print("\n--- Analysis Script Finished ---")