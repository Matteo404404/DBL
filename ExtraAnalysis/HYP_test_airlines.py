import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# Config
airlines = {
    "lufthansa": "metrics_lufthansa.csv",
    "airfrance": "metrics_airfrance.csv",
    "britishairways": "metrics_britishairways.csv"
}
airline_a = "lufthansa"
airline_b = "airfrance"

# Load data
df_a = pd.read_csv(airlines[airline_a])
df_b = pd.read_csv(airlines[airline_b])

# influencers as75th percentile
for df in [df_a, df_b]:
    threshold = df['root_follower_count'].quantile(0.75)
    df['is_influencer'] = (df['root_follower_count'] >= threshold).astype(int)
    df['is_positive_closure'] = (df['sentiment_change_after_airline'] > 0).astype(int)

# tests
ttest_metrics = [
    "avg_airline_sentiment",
    "sentiment_change_after_airline",
    "avg_airline_response_time"
]

ztest_metrics = {
    "did_airline_reply": "Reply Rate",
    "is_influencer": "Influencer Rate (Bias)",
    "is_positive_closure": "Positive Closure Rate"
}

follower_metric = "root_follower_count"

# T results
t_results = []
for metric in ttest_metrics:
    a, b = df_a[metric].dropna(), df_b[metric].dropna()
    if len(a) < 2 or len(b) < 2:
        continue
    stat, p = ttest_ind(a, b, equal_var=False)
    t_results.append({
        "Metric": metric,
        f"{airline_a} Mean": round(a.mean(), 3),
        f"{airline_b} Mean": round(b.mean(), 3),
        "p-value": round(p, 4),
        "Significant (p<.05)": "Yes" if p < 0.05 else "No"
    })
print("\nT-Test Results:\n")
print(pd.DataFrame(t_results).to_string(index=False))

# Z proportions test results
z_results = []
for binary_col, label in ztest_metrics.items():
    count = [df_a[binary_col].sum(), df_b[binary_col].sum()]
    nobs = [len(df_a), len(df_b)]
    stat, p = proportions_ztest(count, nobs)
    z_results.append({
        "Metric": label,
        f"{airline_a} Rate": f"{count[0]}/{nobs[0]} = {count[0]/nobs[0]:.3f}",
        f"{airline_b} Rate": f"{count[1]}/{nobs[1]} = {count[1]/nobs[1]:.3f}",
        "p-value": round(p, 4),
        "Significant (p<.05)": "Yes" if p < 0.05 else "No"
    })
print("\nZ-Test Results:\n")
print(pd.DataFrame(z_results).to_string(index=False))

# favoritism test
replied_a = df_a[df_a['did_airline_reply'] == 1]['root_follower_count']
replied_b = df_b[df_b['did_airline_reply'] == 1]['root_follower_count']
if len(replied_a) > 1 and len(replied_b) > 1:
    stat, p = ttest_ind(replied_a, replied_b, equal_var=False)
    print(f"\nInfluencer Favoritism (Reply Follower Count)\n"
          f"{airline_a} mean: {replied_a.mean():.1f}, {airline_b} mean: {replied_b.mean():.1f}, p = {p:.4f} "
          f"=> {'Significant' if p < 0.05 else 'Not significant'}")
