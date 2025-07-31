from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MONGO_URI = "mongodb+srv://ydandriyal:Zeus_4321@twitter-db.gcbk8ct.mongodb.net/?retryWrites=true&w=majority&appName=twitter-db"
DB_NAME = "MYDB"

# connect
client = MongoClient(MONGO_URI)
users = client[DB_NAME]['users']

# fech followers count for all users
cursor = users.find({}, {'followers_count': 1, '_id': 0})
df = pd.DataFrame(list(cursor))

# summ stats
desc = df['followers_count'].describe(percentiles=[.5, .75, .9, .95, .99])
print(desc)


# pick  influencer cutoff
influencer_cutoff = desc['75%']

# label each user
df['is_influencer'] = df['followers_count'] >= influencer_cutoff

# visualize histogram
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(df['followers_count'], bins=50, log=True)
plt.axvline(influencer_cutoff, color='red', linestyle='--',
            label=f'90th pct = {int(influencer_cutoff):,}')
plt.title('Histogram of follower counts (log‐scale)')
plt.xlabel('Followers')
plt.legend()

plt.subplot(1,2,2)
counts, bin_edges = np.histogram(df['followers_count'], bins=100, density=True)
cdf = np.cumsum(counts) * np.diff(bin_edges)
plt.plot(bin_edges[1:], cdf)
plt.axhline(0.90, color='red', linestyle='--')
plt.title('CDF of follower counts')
plt.xlabel('Followers')
plt.ylabel('Cumulative %')
plt.tight_layout()
plt.show()
