import json
import os
import time
import csv

RAW_DATA_DIR    = "/home/yash/Downloads/data"
CLEAN_DATA_DIR  = "Data/Clean_Data"
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# For overall summary
total_raw     = 0
total_clean   = 0
total_dropped = 0

metrics = []

def read_reduce_file(file_path: str, output_path: str):
    global total_raw, total_clean, total_dropped

    raw_count = 0
    out_list  = []

    # measure raw file size in MB
    in_size_mb = os.path.getsize(file_path) / (1024 ** 2)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_count += 1
            if line.strip() == 'Exceeded connection limit for user':
                continue
            try:
                res = json.loads(line)
                created_at = res['created_at'].replace('+0000 ', '')
                tweet_id   = int(res['id'])
                text = res.get('extended_tweet', {}).get('full_text', res['text']) if res['truncated'] else res['text']
                tweet_data = {
                    'in_reply_to_status_id': res['in_reply_to_status_id'],
                    'in_reply_to_user_id': res['in_reply_to_user_id'],
                    'in_reply_to_screen_name': res['in_reply_to_screen_name'],
                    'is_quote_status': res['is_quote_status'],
                    'quoted_status_id': res.get('quoted_status_id'),
                    'reply_count': res.get('reply_count', 0),
                    'retweet_count': res.get('retweet_count', 0),
                    'favorite_count': res.get('favorite_count', 0),
                    'lang': res['lang']
                }
                user = {
                    'user_id': res['user']['id'],
                    'screen_name': res['user']['screen_name'],
                    'name': res['user']['name'],
                    'followers_count': res['user']['followers_count'],
                    'friends_count': res['user']['friends_count']
                }
                try:
                    place = {
                        'country_code': res['place']['country_code'],
                        'country': res['place']['country'],
                        'city': res['place']['name']
                    }
                except:
                    place = None
                entities = res.get('entities', {})
                if entities.get('hashtags'):
                    entities['hashtags'] = [h['text'] for h in entities['hashtags']]
                entities['urls'] = [
                    u.get('expanded_url') for u in entities.get('urls', []) if u.get('expanded_url')
                ]
                for m in entities.get('user_mentions', []):
                    m.pop('id_str', None); m.pop('indices', None); m.pop('name', None)
                entities.pop('symbols', None); entities.pop('media', None)
                media = None
                if 'extended_entities' in res and 'media' in res['extended_entities']:
                    ml = res['extended_entities']['media']
                    media = {'media_type': ml[0].get('type'), 'media_count': len(ml)}

                final = {
                    'created_at': created_at,
                    'id': tweet_id,
                    'text': text,
                    'tweet_data': tweet_data,
                    'user': user,
                    'place': place,
                    'entities': entities,
                    'media': media
                }
                out_list.append(json.dumps(final))
            except Exception:
                continue

    # write cleaned file
    clean_filename = os.path.basename(file_path).replace("airlines", "airlines-c")
    clean_path = os.path.join(output_path, clean_filename)
    with open(clean_path, 'w', encoding='utf-8') as out_f:
        for obj in out_list:
            out_f.write(obj + '\n')

    clean_count = len(out_list)
    dropped     = raw_count - clean_count
    out_size_mb = os.path.getsize(clean_path) / (1024 ** 2)

    # update totals
    total_raw     += raw_count
    total_clean   += clean_count
    total_dropped += dropped

    # record metrics
    metrics.append({
        'file': os.path.basename(file_path),
        'raw_tweets': raw_count,
        'clean_tweets': clean_count,
        'dropped': dropped,
        'pct_dropped': f"{dropped/raw_count:.1%}" if raw_count else "N/A",
        'size_before_mb': round(in_size_mb, 2),
        'size_after_mb':  round(out_size_mb, 2),
        'size_reduction_mb': round(in_size_mb - out_size_mb, 2),
        'pct_reduction': f"{(in_size_mb - out_size_mb)/in_size_mb:.1%}" if in_size_mb else "N/A"
    })

    # per-file log
    print(f"\n {os.path.basename(file_path)}")
    print(f"   Raw tweets:     {raw_count:,}")
    print(f"   Clean tweets:   {clean_count:,}")
    print(f"   Dropped tweets: {dropped:,} ({metrics[-1]['pct_dropped']})")
    print(f"   Size before:    {in_size_mb:.1f} MB")
    print(f"   Size after:     {out_size_mb:.1f} MB")
    print(f"   Size reduction: {in_size_mb - out_size_mb:.1f} MB ({metrics[-1]['pct_reduction']})")

    return clean_count, clean_path

def read_reduce_directory(directory_path: str, output_dir: str):
    files = sorted(os.listdir(directory_path))
    for i, fn in enumerate(files, 1):
        if not fn.startswith("airlines-") or fn.startswith("airlines-c"):
            continue
        path = os.path.join(directory_path, fn)
        start = time.time()
        read_reduce_file(path, output_dir)
        print(f"   -> Done in {time.time()-start:.1f}s ({i}/{len(files)})\n")

def write_metrics_csv():
    csv_path = os.path.join(CLEAN_DATA_DIR, "cleaning_debug_log.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\n Metrics written to {csv_path}")

if __name__ == "__main__":
    read_reduce_directory(RAW_DATA_DIR, CLEAN_DATA_DIR)

    # overall summary
    print("\n=== OVERALL SUMMARY ===")
    print(f"Total raw tweets:    {total_raw:,}")
    print(f"Total clean tweets:  {total_clean:,}")
    print(f"Total dropped:       {total_dropped:,} ({total_dropped/total_raw:.1%})")

    # write CSV of per-file logs
    if metrics:
        write_metrics_csv()
