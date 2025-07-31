import os
import psycopg2
import json
import logging
from datetime import datetime
import time

# Database connection details
DB_HOST = "localhost"
DB_NAME = "mydb"
DB_USER = "postgres"
DB_PASSWORD = "vectorisiwhatever"

# Raw data folder
FOLDER_PATH = r'C:\Users\matei\Desktop\DBL\DBL\Cleaned Data\Cleaned'

# Lufthansa's Twitter account ID
LUFTHANSA_ID = 124476322

# Preview Mode
LOG_PROGRESS_EVERY_N = 1000
PREVIEW_MODE = False
PREVIEW_LIMIT = 500 if PREVIEW_MODE else float('inf')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_db():
    logging.info(f"Connecting to database '{DB_NAME}' on host '{DB_HOST}' as user '{DB_USER}'...")
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def process_file(file_path, cursor, preview_limit):
    objects_processed = 0
    errors_in_file = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = json.load(f)
            lines = content if isinstance(content, list) else content.strip().split('\n')
        except json.JSONDecodeError:
            lines = f.read().strip().split('\n')

    for i, obj_str in enumerate(lines):
        if objects_processed >= preview_limit:
            logging.info(f"Reached preview limit of {preview_limit}. Stopping early.")
            break

        try:
            data = obj_str if isinstance(obj_str, dict) else json.loads(obj_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error at line {i+1}: {e}")
            errors_in_file += 1
            continue

        user_data = data.get("user", {})
        text_lower = data.get("text", "").lower()
        entities = data.get("entities", {})

        is_official_lufthansa = (user_data.get("user_id") == LUFTHANSA_ID)
        mentions_lufthansa_text = "lufthansa" in text_lower
        mentions_lufthansa_hashtag = any("lufthansa" in h.get("text", "").lower() for h in entities.get("hashtags", []))
        mentions_lufthansa_user = any("lufthansa" in m.get("screen_name", "").lower() for m in entities.get("user_mentions", []))

        if not (is_official_lufthansa or mentions_lufthansa_text or mentions_lufthansa_hashtag or mentions_lufthansa_user):
            continue

        print(f"Lufthansa-related: {data.get('text')}")

        try:
            # table
            user_id = user_data.get("user_id")
            user_name = user_data.get("name")
            screen_name = user_data.get("screen_name")
            followers_count = user_data.get("followers_count")
            friends_count = user_data.get("friends_count")
            statuses_count = user_data.get("statuses_count")
            verified = user_data.get("verified")
            protected = user_data.get("protected")
            profile_image_url = user_data.get("profile_image_url")

            cursor.execute("""
                INSERT INTO users (user_id, name, screen_name, followers_count, friends_count,
                                   statuses_count, verified, protected, profile_image_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING;
            """, (user_id, user_name, screen_name, followers_count, friends_count,
                  statuses_count, verified, protected, profile_image_url))

            # technical table
            tweet_id = data.get("id")
            created_at_str = data.get("created_at")
            created_at_dt = None
            if created_at_str:
                try:
                    created_at_dt = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S %z %Y')
                except (ValueError, TypeError):
                    logging.warning(f"Invalid date format for tweet ID {tweet_id}: {created_at_str}")

            text = data.get("text")
            lang = data.get("lang")
            truncated = data.get("truncated")

            tweet_data = data.get("tweet_data", {})
            in_reply_to_status_id = tweet_data.get("in_reply_to_status_id")
            in_reply_to_user_id = tweet_data.get("in_reply_to_user_id")
            in_reply_to_screen_name = tweet_data.get("in_reply_to_screen_name")
            is_quote_status = tweet_data.get("is_quote_status")
            quoted_status_id = tweet_data.get("quoted_status_id")
            place = data.get("place")

            cursor.execute("""
                INSERT INTO tweet_technical (tweet_id, created_at, text, lang, truncated,
                                             in_reply_to_status_id, in_reply_to_user_id,
                                             in_reply_to_screen_name, is_quote_status,
                                             quoted_status_id, place, user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING;
            """, (tweet_id, created_at_dt, text, lang, truncated,
                  in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name,
                  is_quote_status, quoted_status_id, place, user_id))

            # entities
            media = data.get("media", {})
            media_type = media.get("media_type")
            media_count = media.get("media_count")

            urls = entities.get("urls", [])
            url_count = len(urls) if isinstance(urls, list) else urls if isinstance(urls, int) else 0
            hashtag_count = len(entities.get("hashtags", []))
            user_mention_count = len(entities.get("user_mentions", []))

            cursor.execute("""
                INSERT INTO entities (tweet_id, media_type, media_count,
                                      url_count, hashtag_count, user_mention_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING;
            """, (tweet_id, media_type, media_count,
                  url_count, hashtag_count, user_mention_count))

            objects_processed += 1

            if (i + 1) % LOG_PROGRESS_EVERY_N == 0:
                logging.info(f"Processed {i + 1}/{len(lines)} lines...")

        except Exception as e:
            logging.error(f"Unexpected error at line {i+1}: {type(e).__name__} - {e}")
            errors_in_file += 1
            continue

    return objects_processed, errors_in_file

def main():
    conn = None
    cursor = None
    start_time_total = time.time()
    total_records_committed = 0
    total_files_processed = 0

    try:
        conn = connect_db()
        cursor = conn.cursor()

        logging.info(f"Scanning folder: {FOLDER_PATH}")
        for filename in os.listdir(FOLDER_PATH):
            if filename.endswith('.json'):
                file_path = os.path.join(FOLDER_PATH, filename)
                logging.info(f"Starting file: {filename}")

                start_time_file = time.time()
                objects_processed, errors_in_file = process_file(file_path, cursor, PREVIEW_LIMIT)

                conn.commit()

                elapsed_file = time.time() - start_time_file
                logging.info(f"Finished {filename}: Inserted {objects_processed} Lufthansa tweets, {errors_in_file} errors in {elapsed_file:.1f}s")

                total_records_committed += objects_processed
                total_files_processed += 1

    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {type(e).__name__} - {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
        end_time_total = time.time()
        logging.info(f"--- Script Finished ---")
        logging.info(f"Total files processed: {total_files_processed}")
        logging.info(f"Total Lufthansa records committed: {total_records_committed}")
        logging.info(f"Total time: {end_time_total - start_time_total:.1f}s")

if __name__ == "__main__":
    main()
