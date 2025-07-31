import os
import pandas as pd
import plotly.express as px

def create_airline_comparison_visuals(df: pd.DataFrame, airlines_enabled: dict, out_dir="airline_comparison_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    airline_colors = {
        "lufthansa": "#1f77b4",
        "airfrance": "#ff7f0e",
        "britishairways": "#2ca02c"
    }

    enabled_airlines = [airline for airline, enabled in airlines_enabled.items() if enabled]
    df = df[df['airline'].isin(enabled_airlines)]

    figs = {}

    df = df[df['airline'].notna()].copy()
    df_full = df.copy()

    df['avg_airline_response_time_min'] = df['avg_airline_response_time'] / 60
    df = df[df['avg_airline_response_time_min'] < 180]  # remove extreme response times

    # Violin: Response Time
    fig_response_time = px.violin(
        df, x='airline', y='avg_airline_response_time_min',
        color='airline', box=True, points='outliers',
        color_discrete_map=airline_colors,
        title="Distribution of Airline Response Times (Minutes)",
        labels={'avg_airline_response_time_min': "Avg Response Time (min)"}
    )
    fig_response_time.update_yaxes(range=[0, 180])
    figs['response_time'] = fig_response_time

    # Violin: Sentiment Change
    fig_sentiment = px.violin(
        df, x='airline', y='sentiment_change_after_airline',
        color='airline', box=True, points='outliers',
        color_discrete_map=airline_colors,
        title="Distribution of Sentiment Change After Airline Response",
        labels={'sentiment_change_after_airline': "Sentiment Change"}
    )
    fig_sentiment.update_yaxes(range=[-1, 1])
    figs['sentiment_change'] = fig_sentiment

    # Bar Chart: Reply Rate by Airline (on unfiltered full data)
    reply_rate_df = df_full.groupby('airline')['did_airline_reply'].mean().reset_index()
    fig_reply_rate = px.bar(
        reply_rate_df, x='airline', y='did_airline_reply',
        color='airline', color_discrete_map=airline_colors,
        title="Reply Rate by Airline",
        labels={'did_airline_reply': "Reply Rate"}
    )
    fig_reply_rate.update_yaxes(range=[0, 0.25])
    figs['reply_rate'] = fig_reply_rate

    # Bar Chart: Avg Follower Count of Replied Users
    influencer_df = df_full[df_full['did_airline_reply'] == True]
    mean_follower_df = influencer_df.groupby('airline')['root_follower_count'].mean().reset_index()
    fig_follower_bias = px.bar(
        mean_follower_df, x='airline', y='root_follower_count',
        color='airline', color_discrete_map=airline_colors,
        title="Avg Follower Count of Replied Users",
        labels={'root_follower_count': 'Avg Root Follower Count'}
    )
    figs['influencer_bias'] = fig_follower_bias

    # Save plots
    for name, fig in figs.items():
        fig.write_html(os.path.join(out_dir, f"{name}.html"))
        print(f"Saved: {name}.html")

    return figs


if __name__ == "__main__":
    airlines_enabled = {
        "lufthansa": True,
        "airfrance": True,
        "britishairways": False
    }

    dfs = []
    if airlines_enabled["lufthansa"]:
        df_lh = pd.read_csv("metrics_lufthansa.csv")
        df_lh['airline'] = 'lufthansa'
        dfs.append(df_lh)

    if airlines_enabled["airfrance"]:
        df_af = pd.read_csv("metrics_airfrance.csv")
        df_af['airline'] = 'airfrance'
        dfs.append(df_af)

    if airlines_enabled["britishairways"]:
        df_ba = pd.read_csv("metrics_britishairways.csv")
        df_ba['airline'] = 'britishairways'
        dfs.append(df_ba)

    df_all = pd.concat(dfs, ignore_index=True)
    create_airline_comparison_visuals(df_all, airlines_enabled)
