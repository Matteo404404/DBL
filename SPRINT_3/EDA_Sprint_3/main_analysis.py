from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
import json
import os
import re
from collections import Counter 
from transformers import pipeline 
import torch 
from pathlib import Path

#config
base_dir = Path(__file__).resolve().parent.parent.parent
MONGO_URI = "mongodb+srv://mateiberari:matei2005@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
DB_NAME = "MYDB"

AIRLINE_CONFIGS = {
    "Lufthansa": {
        "threads_collection_names": ["threads_lufthansa", "threads_new"],
        "user_id": 124476322,
        "output_dir_suffix": "lufthansa",
        "name_for_stopwords": "lufthansa"
    },
    "BritishAirways": {
        "threads_collection_names": ["threads_british_airways"],
        "user_id": 18332190,
        "output_dir_suffix": "british_airways",
        "name_for_stopwords": "british airways ba"
    }
    # Add other airlines when ready
}

# inizialize zero-shot classifier
ZERO_SHOT_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
print(f"Attempting to load zero-shot model: {ZERO_SHOT_MODEL_NAME}...")
device_to_use = -1
if torch.cuda.is_available():
    try:
        torch.cuda.get_device_name(0) 
        device_to_use = 0 
        print(f"GPU (CUDA) is available. Using device: {torch.cuda.get_device_name(device_to_use)}")
    except Exception as e:
        print(f"Could not initialize CUDA device (Error: {e}). Falling back to CPU.")
        device_to_use = -1
else:
    print("GPU (CUDA) not available, using CPU for zero-shot classification.")

try:
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=ZERO_SHOT_MODEL_NAME,
        tokenizer=ZERO_SHOT_MODEL_NAME,
        use_fast=False,  
        device=device_to_use
    )
    print(f"Zero-shot classification pipeline loaded successfully.")
except Exception as e:
    print(f"CRITICAL WARNING: Could not load zero-shot classification model: {e}")
    print("Will proceed with keyword-based labeling only if zero-shot fails.")
    zero_shot_classifier = None

CANDIDATE_PROBLEM_LABELS_FOR_ZERO_SHOT = [
    "flight cancellation or rebooking", "flight delay or missed connection",
    "lost or damaged luggage", "refund or compensation request",
    "booking, ticketing, or seat issue", "customer service complaint",
    "in-flight experience (food, entertainment, wifi, crew)",
    "loyalty program or miles issue", "website or app technical problem",
    "general question or information request", "baggage allowance or fee query",
    "pet travel inquiry or issue", "positive feedback or compliment",
    "neutral observation or general mention", "other feedback or unspecified issue"
]

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("MongoDB connection established successfully.")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit()

# parsing
def parse_twitter_timestamp(timestamp_str):
    if not timestamp_str: return None
    try:
        dt_object = datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y')
        return dt_object.replace(tzinfo=timezone.utc)
    except ValueError: return None
print("Helper functions defined.")

# -keyword-based problem labeling function-
PRIMARY_PROBLEM_KEYWORDS = {
    # Most Critical
    "lost_luggage": [
        "lost bag", "lost my bag", "bag lost", "lost luggage", "luggage lost", "missing bag", "missing baggage", 
        "baggage didn't arrive", "suitcase missing", "no bag", "where is my bag", "where is my luggage",
        "baggage claim issue", "pirb", "worldtracer file reference"
    ],
    "flight_cancellation": [
        "flight cancelled", "my flight was cancelled", "cancellation of flight", "cancel my flight",
        "they cancelled", "got cancelled", "flight got cancelled", "what to do cancelled"
    ],
    "flight_delay_significant": [ # For more significant delays or consequences
        "major delay", "long delay", "flight delayed hours", "delayed by hours", "stuck due to delay",
        "missed connection", "missed my connecting flight", "consequences of delay", "overnight delay"
    ],
    "denied_boarding": [
        "denied boarding", "not allowed to board", "overbooked", "oversold flight", "bumped off flight"
    ],
    "refund_issues": [ # Specifically about getting money back
        "refund request", "no refund", "waiting for refund", "issue with refund", "get my money back",
        "reimbursement denied", "claim rejected", "owed a refund", "refund status", "wheres my refund"
    ],

    # Service problems
    "customer_service_complaint": [
        "customer service", "customer support", "helpdesk", "call center", "long hold time", "waiting time",
        "rude agent", "unhelpful staff", "staff attitude", "poor service", "bad service", "terrible service",
        "complaint handling", "ignored", "no response", "unprofessional", "lack of empathy", "misinformed"
    ],
    "booking_ticketing_errors": [
        "booking problem", "ticket issue", "reservation error", "payment failed", "booking confirmation not received",
        "wrong name on ticket", "can't complete booking", "double charged", "fare error", "price mistake"
    ],
    "checkin_boarding_gate_issues": [
        "check-in problem", "online checkin failed", "app checkin error", "kiosk not working",
        "boarding pass issue", "couldn't get boarding pass", "gate agent issue", "boarding chaos", "late boarding",
        "seat assignment problem at gate", "priority boarding issue"
    ],
    "damaged_luggage": [
        "damaged bag", "damaged luggage", "broken suitcase", "bag ripped", "wheel broken", "handle broken",
        "items damaged", "contents missing" # Could also be theft, but often part of damage claim
    ],
    "flight_delay_general": [ # More general delays, less severe than significant
        "flight delay", "delayed", "running late", "flight was late", "late departure", "late arrival",
        "minor delay", "slight delay"
    ],
    "rebooking_issues": [ # Issues specifically with the rebooking process after disruption
        "rebook", "rebooked", "rebooking failed", "difficult to rebook", "no options to rebook",
        "alternative flight", "flight change assistance"
    ],

    # on-board experience
    "seat_comfort_issues": [
        "seat uncomfortable", "broken seat", "seat doesn't recline", "no legroom", "cramped seat",
        "seat assignment complaint", "stuck in middle seat"
    ],
    "food_beverage_complaint": [
        "food quality", "bad meal", "no meal option", "special meal wrong", "cold food",
        "ran out of food", "drink selection poor", "no water offered", "catering issue"
    ],
    "in_flight_entertainment_wifi": [
        "ife not working", "entertainment system broken", "no movies", "screen blank", "headset issue",
        "wifi not working", "no internet", "slow wifi", "wifi cost", "connect to wifi"
    ],
    "cabin_crew_service": [ # Service specifically by cabin crew
        "cabin crew", "flight attendant", "rude flight attendant", "unhelpful crew", "attentive crew",
        "crew service poor", "great crew service" # Could be positive too, but context is problem labeling
    ],
    "cabin_environment": [ # Cleanliness, temperature etc.
        "dirty cabin", "cabin unclean", "messy plane", "toilet dirty",
        "too cold", "too hot", "cabin temperature", "bad smell"
    ],
    "baggage_fees_allowance": [ # Questions or complaints about rules/costs
        "baggage fee", "extra bag cost", "overweight baggage fee", "carry-on policy", "hand luggage size",
        "checked bag limit", "sports equipment fee", "pet fee"
    ],

    # loyalty
    "loyalty_program_issues": [
        "miles missing", "points not credited", "frequent flyer account", "status recognition",
        "upgrade issue", "award ticket problem", "senator lounge", "hon circle service"
    ],
    "website_app_technical": [
        "website error", "app not working", "online system down", "technical issue site",
        "login problem", "password reset failed", "can't access booking", "page frozen"
    ],

    # general
    "pet_travel_query": [
        "pet in cabin", "dog on flight", "cat travel policy", "animal transport rules"
    ],
    "general_query_information": [ # Broad questions, often neutral until an issue arises
        "question about", "information on", "how do i", "can i bring", "what is the policy", "enquiry",
        "details needed", "ask about"
    ],
    # "other_unknown" will be the fallback by the function
}
def get_primary_problem_label_keywords(root_node_text):
    text_lower = str(root_node_text).lower()
    for label, keywords in PRIMARY_PROBLEM_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(str(keyword).lower()) + r'\b'
            if re.search(pattern, text_lower): return label
    return "other_unknown"
print("Keyword-based problem labeling function defined.")
    
# zero-shot problem labeling function
def get_problem_label_zero_shot(tweet_text, candidate_labels, classifier_pipeline, confidence_threshold=0.30): 
    if not classifier_pipeline: return "other_unknown_classifier_unavailable"
    if not tweet_text or not candidate_labels: return "other_unknown_no_input"
    tweet_text = str(tweet_text).strip()
    if len(tweet_text) < 10: return "other_unknown_too_short"
    try:
        result = classifier_pipeline(tweet_text, candidate_labels, multi_label=False)
        if result and result['labels'] and result['scores']:
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            if top_score >= confidence_threshold:
                return top_label.lower().replace(" ", "_").replace("/", "_or_")
            else:
                return f"other_unknown_low_confidence_{top_score:.2f}"
        return "other_unknown_classification_failed"
    except Exception: return "other_unknown_exception"
print("Zero-shot problem labeling function defined.")


# enrichment function for conversation tree nodes
def enrich_conversation_tree_node(node, current_airline_user_id, parent_node_data=None):
    if not node or 'id' not in node: return
    sentiment_obj = node.get('sentiment', {})
    contextual_score = sentiment_obj.get('score')
    node['contextual_sentiment_score'] = contextual_score if contextual_score is not None else 0.0
    node['user_id_author'] = node.get('user_ref')
    node['full_text'] = node.get('text', '')
    if node.get('user_id_author') == current_airline_user_id: node['author_type'] = "Airline"
    elif node.get('user_id_author') is not None: node['author_type'] = "User"
    else: node['author_type'] = "Unknown"
    node['timestamp_dt'] = parse_twitter_timestamp(node.get('time'))
    node['cis'] = 0.0
    node['airline_response_time_seconds'] = None
    if parent_node_data and parent_node_data.get('contextual_sentiment_score') is not None:
        parent_sentiment = parent_node_data.get('contextual_sentiment_score', 0.0)
        current_sentiment = node.get('contextual_sentiment_score', 0.0)
        node['cis'] = current_sentiment - parent_sentiment
    if (node.get('author_type') == "Airline" and parent_node_data and
        parent_node_data.get('author_type') == "User" and node.get('timestamp_dt') and
        parent_node_data.get('timestamp_dt')):
        delta = node['timestamp_dt'] - parent_node_data['timestamp_dt']
        node['airline_response_time_seconds'] = delta.total_seconds()
    else: node['airline_response_time_seconds'] = None
    current_node_as_parent_data = {
        'author_type': node.get('author_type'), 'timestamp_dt': node.get('timestamp_dt'),
        'contextual_sentiment_score': node.get('contextual_sentiment_score')}
    for child_node in node.get("children", []):
        enrich_conversation_tree_node(child_node, current_airline_user_id, parent_node_data=current_node_as_parent_data)
print("Enrichment function 'enrich_conversation_tree_node' defined.")


# analysis functions
def find_ineffective_airline_replies(enriched_tree_root, neg_sentiment_threshold=-0.1, neg_cis_threshold=-0.1):
    problematic_sequences = []
    thread_id_for_report = enriched_tree_root.get('id') if enriched_tree_root else None
    root_labels = enriched_tree_root.get('root_problem_label', "N/A") 
    def _traverse(current_node, parent_node=None, grandparent_node=None):
        if not current_node: return
        if (grandparent_node and parent_node and
            grandparent_node.get('author_type') == "User" and
            parent_node.get('author_type') == "Airline" and 
            current_node.get('author_type') == "User"):
            u2_still_negative = current_node.get('contextual_sentiment_score', 0.0) <= neg_sentiment_threshold
            u2_worsened_conversation = current_node.get('cis', 0.0) <= neg_cis_threshold
            if u2_still_negative or u2_worsened_conversation:
                details = {
                    "thread_root_tweet_id": thread_id_for_report, "root_problem_label": root_labels,
                    "initial_user_tweet_id": grandparent_node.get('id'), "initial_user_text": grandparent_node.get('full_text', '')[:150],
                    "initial_user_sentiment": grandparent_node.get('contextual_sentiment_score'),
                    "airline_reply_id": parent_node.get('id'), "airline_reply_text": parent_node.get('full_text', '')[:150],
                    "airline_response_time_sec": parent_node.get('airline_response_time_seconds'),
                    "airline_reply_sentiment": parent_node.get('contextual_sentiment_score'),
                    "final_user_tweet_id": current_node.get('id'), "final_user_text": current_node.get('full_text', '')[:150],
                    "final_user_sentiment_score": current_node.get('contextual_sentiment_score'), "final_user_cis": current_node.get('cis'),
                    "reason": []}
                if u2_still_negative: details["reason"].append("User reply still negative")
                if u2_worsened_conversation: details["reason"].append("User reply worsened conversation (negative CIS)")
                problematic_sequences.append(details)
        for child_node in current_node.get("children", []):
            _traverse(child_node, parent_node=current_node, grandparent_node=parent_node)
    if enriched_tree_root: _traverse(enriched_tree_root)
    return problematic_sequences
print("Analysis function 'find_ineffective_airline_replies' defined.")

def find_unanswered_negative_user_tweets(enriched_tree_root, neg_sentiment_threshold=-0.5, max_wait_hours=4):
    unanswered_neg_tweets = []
    max_wait_seconds = max_wait_hours * 3600
    thread_id_for_report = enriched_tree_root.get('id') if enriched_tree_root else None
    root_labels = enriched_tree_root.get('root_problem_label', "N/A") 
    def _traverse(node):
        if not node: return
        is_user_tweet = node.get('author_type') == "User"
        is_negative = node.get('contextual_sentiment_score', 0.0) <= neg_sentiment_threshold
        node_timestamp = node.get('timestamp_dt')
        if is_user_tweet and is_negative and node_timestamp:
            airline_replied_in_time = False
            for child in node.get("children", []):
                child_timestamp = child.get('timestamp_dt')
                if child.get('author_type') == "Airline" and child_timestamp:
                    time_delta_seconds = (child_timestamp - node_timestamp).total_seconds()
                    if 0 <= time_delta_seconds <= max_wait_seconds:
                        airline_replied_in_time = True; break
            if not airline_replied_in_time:
                unanswered_neg_tweets.append({
                    "thread_root_tweet_id": thread_id_for_report, "root_problem_label": root_labels,
                    "tweet_id": node.get('id'), "text": node.get('full_text', '')[:200],
                    "user_id": node.get('user_id_author'), "sentiment_score": node.get('contextual_sentiment_score'),
                    "timestamp": node.get('timestamp_dt').isoformat() if node.get('timestamp_dt') else None,
                    "reason": f"Negative user tweet with no Airline reply among direct children within {max_wait_hours}h."})
        for child in node.get("children", []): _traverse(child)
    if enriched_tree_root: _traverse(enriched_tree_root)
    return unanswered_neg_tweets
print("Analysis function 'find_unanswered_negative_user_tweets' defined.")

def find_negative_sentiment_escalations(enriched_tree_root, sharp_decline_cis_threshold=-0.75):
    escalation_points = []
    thread_id_for_report = enriched_tree_root.get('id') if enriched_tree_root else None
    root_labels = enriched_tree_root.get('root_problem_label', "N/A")
    def _traverse(current_node, parent_node=None):
        if not current_node: return
        if parent_node:
            node_cis = current_node.get('cis', 0.0)
            if node_cis <= sharp_decline_cis_threshold:
                escalation_points.append({
                    "thread_root_tweet_id": thread_id_for_report, "root_problem_label": root_labels,
                    "escalating_tweet_id": current_node.get('id'), "escalating_tweet_text": current_node.get('full_text', '')[:200],
                    "escalating_tweet_author_type": current_node.get('author_type'),
                    "escalating_tweet_sentiment": current_node.get('contextual_sentiment_score'),
                    "cis_at_escalation": node_cis,
                    "parent_tweet_id": parent_node.get('id'), "parent_tweet_text": parent_node.get('full_text', '')[:200],
                    "parent_tweet_author_type": parent_node.get('author_type'),
                    "parent_tweet_sentiment": parent_node.get('contextual_sentiment_score'),
                    "timestamp": current_node.get('timestamp_dt').isoformat() if current_node.get('timestamp_dt') else None })
        for child_node in current_node.get("children", []):
            _traverse(child_node, parent_node=current_node)
    if enriched_tree_root: _traverse(enriched_tree_root, parent_node=None)
    return escalation_points
print("Analysis function 'find_negative_sentiment_escalations' defined.")

# processing and saving results
if __name__ == "__main__":
    base_output_dir = base_dir / "JSONs" / "multi_airline_analysis_hybrid_labeling"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created base output directory: {base_output_dir}")

    for airline_name, config in AIRLINE_CONFIGS.items():
        print(f"\n--- Processing Airline: {airline_name} ---")
        current_airline_user_id = config['user_id']
        airline_output_dir_suffix = config['output_dir_suffix']
        airline_specific_output_dir = os.path.join(base_output_dir, airline_output_dir_suffix)
        if not os.path.exists(airline_specific_output_dir):
            os.makedirs(airline_specific_output_dir)

        all_ineffective_replies_airline = []
        all_unanswered_neg_tweets_airline = []
        all_escalations_airline = []
        total_threads_processed_for_airline = 0
        total_threads_skipped_for_airline = 0
        PROCESS_LIMIT_PER_COLLECTION = None 

        for threads_coll_name in config['threads_collection_names']:
            print(f"  Starting to process collection: {threads_coll_name} for {airline_name}...")
            try:
                current_threads_collection = db[threads_coll_name]
                if current_threads_collection.estimated_document_count() == 0:
                    print(f"    Warning: Collection '{threads_coll_name}' is empty. Skipping.")
                    continue
            except Exception as e:
                print(f"    Error accessing collection '{threads_coll_name}': {e}. Skipping.")
                continue

            #  Query only unlabeled documents
            skip_filter = {
                "$or": [
                    {"root_problem_label_derived": {"$exists": False}},
                    {"root_problem_label_derived": None},
                    {"root_problem_label_derived": ""},
                    {"problem_labeling_method_used": {"$exists": False}},
                    {"problem_labeling_method_used": None},
                    {"problem_labeling_method_used": ""}
                ]
            }
            
            # Get counts for reporting
            total_docs_in_collection = current_threads_collection.estimated_document_count()
            unlabeled_docs_count = current_threads_collection.count_documents(skip_filter)
            already_labeled_count = total_docs_in_collection - unlabeled_docs_count
            
            print(f"    Collection '{threads_coll_name}' stats:")
            print(f"      Total documents: {total_docs_in_collection}")
            print(f"      Already labeled: {already_labeled_count}")
            print(f"      Unlabeled (to process): {unlabeled_docs_count}")
            
            if unlabeled_docs_count == 0:
                print(f"    All documents in '{threads_coll_name}' are already labeled. Skipping collection.")
                total_threads_skipped_for_airline += already_labeled_count
                continue

            # Query only unlabeled documents
            thread_cursor = current_threads_collection.find(skip_filter)
            if PROCESS_LIMIT_PER_COLLECTION:
                thread_cursor = thread_cursor.limit(PROCESS_LIMIT_PER_COLLECTION)
            
            collection_threads_count = 0
            collection_threads_skipped = already_labeled_count
            total_threads_skipped_for_airline += collection_threads_skipped
            
            for thread_doc in thread_cursor:
                total_threads_processed_for_airline += 1    
                collection_threads_count += 1
                mongo_doc_id = thread_doc.get('_id')

                if total_threads_processed_for_airline % 200 == 0:
                    print(f"    {airline_name} ({threads_coll_name}): Processing thread {total_threads_processed_for_airline} (Doc ID: {mongo_doc_id})...")

                if "tree" not in thread_doc or not thread_doc["tree"]: 
                    print(f"    Warning: Thread {mongo_doc_id} has no tree structure. Skipping.")
                    continue

                enriched_tree_root_node = thread_doc["tree"]
                enrich_conversation_tree_node(enriched_tree_root_node, current_airline_user_id, parent_node_data=None)
                
                # HYBRID PROBLEM LABELING
                root_problem_label = "other_unknown_default"
                root_labeling_method = "none"
                root_text = enriched_tree_root_node.get('full_text')
                if root_text:
                    label_kw = get_primary_problem_label_keywords(root_text)
                    if label_kw != "other_unknown":
                        root_problem_label = label_kw
                        root_labeling_method = "keyword"
                    elif zero_shot_classifier:
                        label_zs = get_problem_label_zero_shot(
                            root_text, CANDIDATE_PROBLEM_LABELS_FOR_ZERO_SHOT, 
                            zero_shot_classifier, confidence_threshold=0.30
                        )
                        root_problem_label = label_zs
                        root_labeling_method = "zero_shot"
                    else: 
                        root_labeling_method = "keyword_failed_no_zs"
                else:
                    root_problem_label = "N/A_no_root_text"
                
                enriched_tree_root_node['root_problem_label'] = root_problem_label
                enriched_tree_root_node['problem_labeling_method'] = root_labeling_method


                # Update MongoDB with labeling results
                try:
                    current_threads_collection.update_one(
                        {"_id": mongo_doc_id},
                        {"$set": {
                            "root_problem_label_derived": root_problem_label,
                            "problem_labeling_method_used": root_labeling_method,
                            "last_problem_labeled_at": datetime.now(timezone.utc)
                         }})
                except Exception as e:
                    print(f"    Error updating MongoDB for {mongo_doc_id} in {threads_coll_name}: {e}")

                # Continue with analysis functions
                ineffective = find_ineffective_airline_replies(enriched_tree_root_node)
                if ineffective: all_ineffective_replies_airline.extend(ineffective)
                
                unanswered = find_unanswered_negative_user_tweets(enriched_tree_root_node, neg_sentiment_threshold=-0.5)
                if unanswered: all_unanswered_neg_tweets_airline.extend(unanswered)
                
                escalations = find_negative_sentiment_escalations(enriched_tree_root_node, sharp_decline_cis_threshold=-0.75)
                if escalations: all_escalations_airline.extend(escalations)
            
            print(f"    Finished processing collection '{threads_coll_name}':")
            print(f"      New threads processed: {collection_threads_count}")
            print(f"      Threads skipped (already labeled): {collection_threads_skipped}")
        
        print(f"\n  --- Analysis Summary for {airline_name} ---")
        print(f"    Total new threads processed: {total_threads_processed_for_airline}")
        print(f"    Total threads skipped (already labeled): {total_threads_skipped_for_airline}")
        print(f"    Ineffective replies: {len(all_ineffective_replies_airline)}")
        print(f"    Unanswered negative: {len(all_unanswered_neg_tweets_airline)}")
        print(f"    Escalations: {len(all_escalations_airline)}")

        def save_to_json_airline(data, filename, out_dir):
            filepath = os.path.join(out_dir, filename)
            try:
                def convert_datetime(o):
                    if isinstance(o, datetime): return o.isoformat()
                with open(filepath, "w", encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=convert_datetime) 
                print(f"    Results for {airline_name} saved to {filepath}")
            except Exception as e:
                print(f"    Error saving {filename} to {out_dir} for {airline_name}: {e}")
        
        # Save results
        if all_ineffective_replies_airline: 
            save_to_json_airline(all_ineffective_replies_airline, "ineffective_replies.json", airline_specific_output_dir)
        if all_unanswered_neg_tweets_airline: 
            save_to_json_airline(all_unanswered_neg_tweets_airline, "unanswered_negative_tweets.json", airline_specific_output_dir)
        if all_escalations_airline: 
            save_to_json_airline(all_escalations_airline, "negative_escalations.json", airline_specific_output_dir)

    if client: 
        client.close()
        print("\nOverall processing finished.")
        print("Next runs will automatically skip already labeled documents.")