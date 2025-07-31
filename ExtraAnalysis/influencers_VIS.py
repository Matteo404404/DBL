import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


class ConversationVisualizerV2:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'conversation_id' not in self.df.columns:
            self.df['conversation_id'] = self.df.index.astype(str)
        self.prepare_data()

    def prepare_data(self):
        df = self.df

        df['avg_airline_response_time'] = df['avg_airline_response_time'] / 3600

        q75_followers = df['root_follower_count'].quantile(0.75)

        df['influencer_category'] = pd.cut(
            df['root_follower_count'],
            bins=[-1, q75_followers, float('inf')],
            labels=['Regular Users', 'Top 25%'],
            include_lowest=True
        )

        self.q75_threshold = q75_followers

        time_bins_hours = [-1, 5 / 60, 30 / 60, 2, 8, 24, float('inf')]
        time_labels = ['Immediate (<5m)', 'Fast (5-30m)', 'Quick (30m-2h)',
                       'Standard (2-8h)', 'Slow (8-24h)', 'Very Slow (24h+)']

        df['response_time_category'] = pd.cut(
            df['avg_airline_response_time'],
            bins=time_bins_hours,
            labels=time_labels
        )

        df['diversity_score'] = (df['num_unique_users'] / df['num_messages'].replace(0, 1)).clip(0, 1)

        df['engagement_score'] = (
                df['airline_reply_rate'] * 0.3 +
                df['sentiment_change_after_airline'] * 0.3 +
                df['diversity_score'] * 0.2 +
                (1 / (1 + df['avg_airline_response_time'])).clip(0, 1) * 0.2
        )

        min_score = df['engagement_score'].min()
        max_score = df['engagement_score'].max()
        if max_score > min_score:
            df['engagement_score'] = (df['engagement_score'] - min_score) / (max_score - min_score)
        else:
            df['engagement_score'] = 0.5

        self.df = df

    def create_engagement_quadrants(self):
        df = self.df
        x = df['avg_airline_response_time']
        y = df['sentiment_change_after_airline']

        x_limit = x.quantile(0.99)

        # Calculate median
        filtered_data = df[df['avg_airline_response_time'] <= x_limit]
        xm, ym = filtered_data['avg_airline_response_time'].median(), df['sentiment_change_after_airline'].median()

        fig = px.scatter(
            df, x=x, y=y,
            size='airline_reply_rate', color='influencer_category',
            hover_name='conversation_id',
            hover_data=['num_messages', 'airline_reply_rate'],
            title="Effort vs. Impact Quadrants (X-axis clipped at 99th percentile)"
        )
        fig.add_vline(x=xm, line_dash="dash", line_color="gray", annotation_text="Median Response Time")
        fig.add_hline(y=ym, line_dash="dash", line_color="gray", annotation_text="Median Sentiment Impact")

        # Apply the axis limit
        fig.update_xaxes(range=[-1, x_limit * 1.05])

        return fig

    def create_3d_landscape(self):
        df = self.df
        # Ensure size values are positive and handle edge cases
        size_values = np.maximum(df['num_messages'], 1)  # Ensure minimum size of 1
        size_values = np.clip(size_values, 2, 30)  # Clip to have desired range

        fig = go.Figure(data=[go.Scatter3d(
            x=np.log10(df['root_follower_count'] + 1),
            y=df['airline_reply_rate'],
            z=df['sentiment_change_after_airline'],
            mode='markers',
            marker=dict(
                size=size_values,
                color=df['engagement_score'],
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Engagement")
            )
        )])
        fig.update_layout(title="3D Engagement Landscape",
                          scene=dict(
                              xaxis_title='log10(Followers)',
                              yaxis_title='Reply Rate',
                              zaxis_title='Sentiment Change'))
        return fig

    def create_correlation_heatmap(self):
        df = self.df
        cols = ['airline_reply_rate', 'sentiment_change_after_airline', 'engagement_score',
                'avg_airline_response_time', 'num_messages', 'num_unique_users']
        corr = df[cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        return fig

    def create_response_time_sankey(self):
        df = self.df
        counts = df.groupby(['influencer_category', 'response_time_category']).size().reset_index(name='count')
        categories = df['influencer_category'].dropna().unique().tolist()
        times = df['response_time_category'].dropna().unique().tolist()

        labels = [str(cat) for cat in categories] + [str(t) for t in times]
        source = counts['influencer_category'].apply(lambda x: labels.index(str(x)))
        target = counts['response_time_category'].apply(lambda x: labels.index(str(x)))
        values = counts['count']

        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
            link=dict(source=source, target=target, value=values)
        )])
        fig.update_layout(title_text="Response Time Flow by User Category", font_size=10)
        return fig

    def create_radar_fingerprints(self):
        df = self.df
        metrics = ['airline_reply_rate', 'sentiment_change_after_airline', 'engagement_score']
        categories = df['influencer_category'].dropna().unique()

        fig = go.Figure()
        for cat in categories:
            row = df[df['influencer_category'] == cat][metrics].mean()
            fig.add_trace(go.Scatterpolar(r=row.values, theta=metrics, fill='toself', name=str(cat)))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          showlegend=True,
                          title="Radar Comparison: Influencer Categories")
        return fig

    def create_anomaly_plot(self):
        df = self.df
        features = ['root_follower_count', 'airline_reply_rate', 'avg_sentiment', 'sentiment_change_after_airline',
                    'avg_airline_response_time', 'max_depth']
        X = df[features].fillna(0)

        if X.shape[0] < 2:
            return go.Figure().update_layout(title="Not enough data for Anomaly Detection")

        X_scaled = StandardScaler().fit_transform(X)
        X_pca = PCA(n_components=2).fit_transform(X_scaled)

        df['pc1'] = X_pca[:, 0]
        df['pc2'] = X_pca[:, 1]
        df['is_anomaly'] = np.linalg.norm(X_pca, axis=1) > np.percentile(np.linalg.norm(X_pca, axis=1), 95)

        fig = px.scatter(
            df, x='pc1', y='pc2', color='is_anomaly',
            hover_data=features, title="Anomaly Detection (PCA Space)"
        )
        print("Anomaly Distribution:")
        print(df['is_anomaly'].value_counts(normalize=True))
        return fig

    def create_response_time_distribution_by_user_type(self):
        df = self.df

        fig = px.box(
            df,
            x='influencer_category',
            y='avg_airline_response_time',
            color='influencer_category',
            title='Response Time Distribution by User Type',
            labels={
                'influencer_category': 'User Type',
                'avg_airline_response_time': 'Response Time (hours)'
            }
        )
        fig.update_yaxes(type="log", title_text="Response Time (hours, log scale)")
        fig.update_layout(showlegend=False)

        return fig

    def create_influencer_impact_comparison(self):
        df = self.df

        metrics = ['airline_reply_rate', 'sentiment_change_after_airline', 'engagement_score',
                   'num_messages', 'num_unique_users', 'avg_airline_response_time']

        comparison_data = []
        for category in df['influencer_category'].unique():
            if pd.isna(category):
                continue
            subset = df[df['influencer_category'] == category]
            for metric in metrics:
                comparison_data.append({
                    'Category': category,
                    'Metric': metric,
                    'Value': subset[metric].mean(),
                    'Count': len(subset)
                })

        comparison_df = pd.DataFrame(comparison_data)

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Reply Rate', 'Sentiment Change', 'Engagement Score',
                            'Avg Messages', 'Unique Users', 'Response Time (hrs)'],
            vertical_spacing=0.12
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, metric in enumerate(metrics):
            row = (i // 3) + 1
            col = (i % 3) + 1

            metric_data = comparison_df[comparison_df['Metric'] == metric]

            fig.add_trace(
                go.Bar(
                    x=metric_data['Category'],
                    y=metric_data['Value'],
                    name=metric,
                    marker_color=colors,
                    showlegend=False,
                    text=[f"{v:.2f}" for v in metric_data['Value']],
                    textposition='auto'
                ),
                row=row, col=col
            )

        fig.update_layout(
            title="User Type Impact Comparison: Key Metrics by Category",
            height=600,
            showlegend=False
        )

        return fig

    def create_influencer_roi_analysis(self):
        df = self.df
        roi_data = []
        for category in df['influencer_category'].unique():
            if pd.isna(category):
                continue
            subset = df[df['influencer_category'] == category]
            potential_reach = (subset['root_follower_count'] * subset['airline_reply_rate']).mean()
            quality_score = (
                    subset['sentiment_change_after_airline'] * 0.4 +
                    subset['engagement_score'] * 0.3 +
                    (subset['num_unique_users'] / subset['num_messages'].replace(0, 1)).mean() * 0.3
            ).mean()
            roi_data.append({
                'Category': category,
                'Potential_Reach': potential_reach,
                'Quality_Score': quality_score,
                'Avg_Followers': subset['root_follower_count'].mean(),
                'Count': len(subset),
                'Engagement_Rate': subset['airline_reply_rate'].mean()
            })

        roi_df = pd.DataFrame(roi_data)

        # Ensure size values are positive
        size_values = np.maximum(roi_df['Count'], 1)

        fig = px.scatter(
            roi_df,
            x='Potential_Reach',
            y='Quality_Score',
            size=size_values,
            color='Category',
            hover_data=['Avg_Followers', 'Engagement_Rate'],
            title="User Type ROI Analysis: Quality vs Reach",
            labels={
                'Potential_Reach': 'Potential Reach (Followers × Engagement)',
                'Quality_Score': 'Conversation Quality Score'
            }
        )
        fig.add_annotation(
            x=roi_df['Potential_Reach'].max() * 0.7,
            y=roi_df['Quality_Score'].max() * 0.9,
            text="High Quality<br>High Reach<br>Prime Target",
            showarrow=True,
            arrowhead=2,
            bgcolor="rgba(255,255,0,0.3)"
        )
        return fig

    def create_influencer_sentiment_journey(self):
        df = self.df
        categories = [cat for cat in df['influencer_category'].unique() if pd.notna(cat)]

        if len(categories) == 0:
            return go.Figure().update_layout(title="No valid user categories found")

        fig = make_subplots(
            rows=1, cols=len(categories),
            subplot_titles=categories,
            shared_yaxes=True
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, category in enumerate(categories):
            subset = df[df['influencer_category'] == category]
            valid_subset = subset.dropna(subset=['avg_sentiment', 'sentiment_change_after_airline'])

            if len(valid_subset) == 0:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text="No valid sentiment data",
                    xref=f"x{i + 1}", yref=f"y{i + 1}" if i > 0 else "y",
                    showarrow=False
                )
                continue

            color_idx = i % len(colors)
            fig.add_trace(
                go.Scatter(
                    x=valid_subset['avg_sentiment'],
                    y=valid_subset['sentiment_change_after_airline'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors[color_idx],
                        opacity=0.6
                    ),
                    name=category,
                    showlegend=(i == 0),
                    text=[f"Followers: {f:,.0f}<br>Messages: {m}" for f, m in
                          zip(valid_subset['root_follower_count'], valid_subset['num_messages'])],
                    hovertemplate="<b>%{text}</b><br>Initial Sentiment: %{x:.2f}<br>Sentiment Change: %{y:.2f}<extra></extra>"
                ),
                row=1, col=i + 1
            )

            if len(valid_subset) > 1 and valid_subset['avg_sentiment'].std() > 1e-10:
                z = np.polyfit(valid_subset['avg_sentiment'], valid_subset['sentiment_change_after_airline'], 1)
                p = np.poly1d(z)
                x_range = valid_subset['avg_sentiment'].max() - valid_subset['avg_sentiment'].min()
                x_trend = np.linspace(valid_subset['avg_sentiment'].min() - x_range * 0.1,
                                      valid_subset['avg_sentiment'].max() + x_range * 0.1, 50)
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        line=dict(color=colors[color_idx], dash='dash', width=2),
                        name=f'{category} Trend',
                        showlegend=False,
                        hovertemplate="Trend line<extra></extra>"
                    ),
                    row=1, col=i + 1
                )

        fig.update_layout(
            title="Sentiment Journey by User Category",
            height=400,
            showlegend=True
        )
        fig.update_xaxes(title_text="Initial Sentiment")
        fig.update_yaxes(title_text="Sentiment Change After Reply", col=1)

        for i in range(len(categories)):
            fig.add_hline(y=0, line_dash="dot", line_color="lightgray", opacity=0.5, row=1, col=i + 1)
            fig.add_vline(x=0, line_dash="dot", line_color="lightgray", opacity=0.5, row=1, col=i + 1)

        return fig

    def create_influencer_response_priority_matrix(self):
        df = self.df

        df['urgency_score'] = (
                np.log10(df['root_follower_count'] + 1) / 7 * 0.6 +
                np.maximum(0, -df['avg_sentiment']) * 0.4
        )

        df['impact_potential'] = (
                df['airline_reply_rate'] * 0.4 +
                (df['num_unique_users'] / df['num_messages'].replace(0, 1)).clip(0, 1) * 0.3 +
                np.log10(df['root_follower_count'] + 1) / 7 * 0.3
        )

        # Ensure size values are positive again
        size_values = np.maximum(df['root_follower_count'], 1)

        fig = px.scatter(
            df,
            x='urgency_score',
            y='impact_potential',
            color='influencer_category',
            size=size_values,
            hover_data=['avg_sentiment', 'airline_reply_rate', 'num_messages'],
            title="Response Priority Matrix: Urgency vs Impact Potential"
        )

        x_median = df['urgency_score'].median()
        y_median = df['impact_potential'].median()

        fig.add_vline(x=x_median, line_dash="dash", line_color="gray")
        fig.add_hline(y=y_median, line_dash="dash", line_color="gray")

        fig.add_annotation(x=df['urgency_score'].max() * 0.8, y=df['impact_potential'].max() * 0.9,
                           text="High Priority<br>Respond ASAP", bgcolor="rgba(255,0,0,0.3)")
        fig.add_annotation(x=df['urgency_score'].max() * 0.8, y=df['impact_potential'].min() * 1.1,
                           text="Medium Priority<br>Monitor", bgcolor="rgba(255,255,0,0.3)")
        fig.add_annotation(x=df['urgency_score'].min() * 1.1, y=df['impact_potential'].max() * 0.9,
                           text="Opportunity<br>Engage", bgcolor="rgba(0,255,0,0.3)")
        fig.add_annotation(x=df['urgency_score'].min() * 1.1, y=df['impact_potential'].min() * 1.1,
                           text="Low Priority<br>Standard", bgcolor="rgba(128,128,128,0.3)")

        return fig

    def create_influencer_conversion_funnel(self):
        df = self.df
        funnel_data = []
        for category in df['influencer_category'].unique():
            if pd.isna(category):
                continue
            subset = df[df['influencer_category'] == category]
            total_conversations = len(subset)
            replied = len(subset[subset['airline_reply_rate'] > 0])
            positive_sentiment = len(subset[subset['sentiment_change_after_airline'] > 0])
            high_engagement = len(subset[subset['engagement_score'] > 0.7])

            funnel_data.extend([
                {'Category': category, 'Stage': 'Total Conversations', 'Count': total_conversations, 'Percentage': 100},
                {'Category': category, 'Stage': 'Replied', 'Count': replied,
                 'Percentage': (replied / total_conversations) * 100 if total_conversations > 0 else 0},
                {'Category': category, 'Stage': 'Positive Sentiment', 'Count': positive_sentiment,
                 'Percentage': (positive_sentiment / total_conversations) * 100 if total_conversations > 0 else 0},
                {'Category': category, 'Stage': 'High Engagement', 'Count': high_engagement,
                 'Percentage': (high_engagement / total_conversations) * 100 if total_conversations > 0 else 0}
            ])

        funnel_df = pd.DataFrame(funnel_data)

        fig = px.funnel(
            funnel_df,
            x='Percentage',
            y='Stage',
            color='Category',
            title="User Engagement Conversion Funnel"
        )

        return fig

    def create_investment_recommendation_dashboard(self):
        df = self.df

        stats = {}
        total_conversations = len(df)

        for category in df['influencer_category'].unique():
            if pd.isna(category):
                continue
            subset = df[df['influencer_category'] == category]

            stats[category] = {
                'count': len(subset),
                'percentage': len(subset) / total_conversations * 100,
                'avg_followers': subset['root_follower_count'].mean(),
                'avg_engagement': subset['engagement_score'].mean(),
                'avg_sentiment_change': subset['sentiment_change_after_airline'].mean(),
                'response_rate': subset['airline_reply_rate'].mean(),
                'potential_reach': (subset['root_follower_count'] * subset['airline_reply_rate']).mean()
            }

        summary_data = []
        for category, data in stats.items():
            summary_data.append({
                'User Category': category,
                'Count': f"{data['count']} ({data['percentage']:.1f}%)",
                'Avg Followers': f"{data['avg_followers']:,.0f}",
                'Engagement Score': f"{data['avg_engagement']:.3f}",
                'Sentiment Impact': f"{data['avg_sentiment_change']:.3f}",
                'Response Rate': f"{data['response_rate']:.2f}",
                'Potential Reach': f"{data['potential_reach']:,.0f}"
            })

        summary_df = pd.DataFrame(summary_data)

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[summary_df[col] for col in summary_df.columns],
                       fill_color='lavender',
                       align='left'))
        ])

        fig.update_layout(
            title="Investment Recommendation Dashboard: User Categories Performance",
            height=300
        )

        return fig

    def create_fairness_equity_analysis(self):
        df = self.df.copy()

        reply_rates = df.groupby('influencer_category')['airline_reply_rate'].mean().sort_values()

        sorted_rates = np.sort(reply_rates.values)
        n = len(sorted_rates)
        index = np.arange(1, n + 1)
        gini = ((np.sum((2 * index - n - 1) * sorted_rates)) / (n * np.sum(sorted_rates)))

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=reply_rates.index.astype(str),
            y=reply_rates.values,
            marker_color=['#1f77b4', '#ff7f0e']
        ))
        bar_fig.update_layout(
            title=f"Reply Rate Disparity by User Category (Gini: {gini:.3f})",
            xaxis_title="User Category",
            yaxis_title="Average Reply Rate"
        )

        cumulative_response = np.cumsum(sorted_rates) / np.sum(sorted_rates)
        cumulative_population = np.arange(1, n + 1) / n
        lorenz_fig = go.Figure()
        lorenz_fig.add_trace(go.Scatter(
            x=cumulative_population,
            y=cumulative_response,
            mode='lines+markers',
            name='Lorenz Curve'
        ))
        lorenz_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Line of Equality',
            line=dict(dash='dash', color='gray')
        ))
        lorenz_fig.update_layout(
            title="Lorenz Curve of Reply Rate Equity",
            xaxis_title="Cumulative Share of User Categories",
            yaxis_title="Cumulative Share of Replies"
        )

        return {"fairness_bar": bar_fig, "fairness_lorenz": lorenz_fig}

    def create_simple_distributions(self):
        df = self.df
        figs = {}

        for col in ['airline_reply_rate', 'sentiment_change_after_airline', 'engagement_score',
                    'avg_airline_response_time']:
            fig = px.violin(df, y=col, box=True, points="all",
                            title=f"Distribution of {col.replace('_', ' ').title()}")
            figs[f"dist_{col}"] = fig

        return figs


def run_and_export(df: pd.DataFrame, out_dir="viz_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    viz = ConversationVisualizerV2(df)

    figs = {}

    original_plot_functions = {
        "engagement_quadrants": viz.create_engagement_quadrants,
        "3d_landscape": viz.create_3d_landscape,
        "correlation_heatmap": viz.create_correlation_heatmap,
        "response_time_sankey": viz.create_response_time_sankey,
        "radar_fingerprints": viz.create_radar_fingerprints,
        "anomaly_plot": viz.create_anomaly_plot
    }

    influencer_plot_functions = {
        "influencer_impact_comparison": viz.create_influencer_impact_comparison,
        "influencer_roi_analysis": viz.create_influencer_roi_analysis,
        "influencer_sentiment_journey": viz.create_influencer_sentiment_journey,
        "influencer_response_priority_matrix": viz.create_influencer_response_priority_matrix,
        "influencer_conversion_funnel": viz.create_influencer_conversion_funnel,
        "investment_recommendation_dashboard": viz.create_investment_recommendation_dashboard,
        "fairness_equity_analysis": viz.create_fairness_equity_analysis,
        "response_time_by_user_type": viz.create_response_time_distribution_by_user_type
    }

    all_plot_functions = {**original_plot_functions, **influencer_plot_functions}

    for name, func in all_plot_functions.items():
        try:
            result = func()
            if isinstance(result, dict):
                for subname, subfig in result.items():
                    figs[f"{name}_{subname}"] = subfig
            else:
                figs[name] = result
            print(f"Created plot: {name}")
        except Exception as e:
            print(f"Error creating plot {name}: {str(e)}")
            continue

    try:
        figs.update(viz.create_simple_distributions())
        print("Created simple distribution plots")
    except Exception as e:
        print(f"Error creating distribution plots: {str(e)}")

    for name, fig in figs.items():
        try:
            html_path = os.path.join(out_dir, f"{name}.html")
            fig.write_html(html_path)
            print(f"Saved: {html_path}")
        except Exception as e:
            print(f"Error saving {name}: {str(e)}")


if __name__ == "__main__":
    import time

    start_time = time.time()

    airlines = [
        {"name": "lufthansa", "enabled": True, "csv": "metrics_lufthansa.csv"},
        {"name": "airfrance", "enabled": False, "csv": "metrics_airfrance.csv"},
        {"name": "britishairways", "enabled": False, "csv": "metrics_britishairways.csv"}
    ]

    for airline in airlines:
        if not airline["enabled"]:
            continue

        print(f"\nGenerating visualizations for: {airline['name'].capitalize()}")
        try:
            df = pd.read_csv(airline["csv"])
        except FileNotFoundError:
            print(f"CSV not found: {airline['csv']}, skipping.")
            continue

        out_dir = f"viz_outputs_{airline['name']}"
        run_and_export(df, out_dir=out_dir)

    elapsed = time.time() - start_time
    print(f"\nVisualization complete. Total elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} min)")