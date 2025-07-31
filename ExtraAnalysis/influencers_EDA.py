import pandas as pd
from typing import List, Dict, Any

def load_convo_metrics(filepath: str) -> pd.DataFrame:
    """
    Load the main conversation metrics CSV into a DataFrame.
    """
    return pd.read_csv(filepath)

def compute_numeric_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Compute descriptive statistics for all numeric columns of interest:
    - Count, Mean, Median, Std, Min, Max, 25th/75th percentiles, Range, % Missing.
    """
    stats = {}
    for col in cols:
        data = df[col]
        stats[col] = {
            'count': data.count(),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            '25%': data.quantile(0.25),
            '75%': data.quantile(0.75),
            'max': data.max(),
            'range': data.max() - data.min(),
            '% missing': 100 * data.isnull().sum() / len(df)
        }
    return pd.DataFrame(stats).T

def compute_binary_stats(df: pd.DataFrame, col: str) -> dict:
    """
    Compute counts and percentages for a binary/categorical column.
    """
    value_counts = df[col].value_counts(dropna=False)
    n_total = len(df)
    stats = {}
    for val in value_counts.index:
        stats[f'{val}_count'] = value_counts[val]
        stats[f'{val}_percent'] = 100 * value_counts[val] / n_total
    n_missing = df[col].isnull().sum()
    stats['missing_count'] = n_missing
    stats['missing_percent'] = 100 * n_missing / n_total
    return stats

def report_outliers(df: pd.DataFrame, col: str, n: int = 5) -> dict:
    """
    Identify the top and bottom n outliers for a numeric metric.
    """
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    top_n = s.sort_values(ascending=False).head(n)
    bottom_n = s.sort_values(ascending=True).head(n)
    return {
        'top_n_values': top_n.values.tolist(),
        'top_n_indices': top_n.index.tolist(),
        'bottom_n_values': bottom_n.values.tolist(),
        'bottom_n_indices': bottom_n.index.tolist()
    }

def create_summary_table(numeric_stats: pd.DataFrame,
                        binary_stats: Dict[str, Any],
                        outlier_stats: Dict[str, Any],
                        output_csv: str) -> None:
    """
    Combine all computed statistics into a single summary table and save to CSV.
    """
    summary_df = numeric_stats.copy()

    for k, v in binary_stats.items():
        for stat_name, value in v.items():
            row_label = f"{k}_{stat_name}"
            summary_df.loc[row_label] = [value if i == 0 else None for i in range(len(summary_df.columns))]

    for k, v in outlier_stats.items():
        summary_df.loc[f"{k}_top_n_values"] = [str(v['top_n_values'])] + [None] * (len(summary_df.columns) - 1)
        summary_df.loc[f"{k}_bottom_n_values"] = [str(v['bottom_n_values'])] + [None] * (len(summary_df.columns) - 1)

    summary_df.to_csv(output_csv)
    print(f"Summary table saved to {output_csv}")

def run_eda_workflow(input_csv: str, output_csv: str) -> None:
    """
    Load data, compute all stats, combine, and save summary table.
    """
    df = load_convo_metrics(input_csv)
    numeric_cols = [
        'root_follower_count', 'num_messages', 'num_unique_users', 'num_lh_replies',
        'lh_reply_rate', 'avg_sentiment', 'avg_lh_sentiment',
        'sentiment_change_after_lh', 'max_depth', 'avg_lh_response_time'
    ]
    numeric_stats = compute_numeric_stats(df, numeric_cols)

    binary_stats = {'did_lh_reply': compute_binary_stats(df, 'did_lh_reply')}

    outlier_metrics = ['lh_reply_rate', 'num_lh_replies', 'avg_lh_response_time']
    outlier_stats = {col: report_outliers(df, col, n=5) for col in outlier_metrics}

    create_summary_table(numeric_stats, binary_stats, outlier_stats, output_csv)
    print(f"EDA summary table saved to {output_csv}")


if __name__ == "__main__":
    run_eda_workflow(
        input_csv="convo_metrics.csv",
        output_csv="influencer_eda_summary.csv"
    )
