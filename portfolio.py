import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  


class Portfolio:
    @staticmethod
    def create(returns, metrics_df, etf_data, metric='Sharpe', cluster_col='Cluster'):
        # Ensure metric column is numeric and handle NaN values
        metrics_df[metric] = pd.to_numeric(metrics_df[metric], errors='coerce')
        metrics_df = metrics_df.dropna(subset=[metric, cluster_col])

        # Select top strategies within each cluster based on the specified metric
        top_strategies = metrics_df.loc[metrics_df.groupby(cluster_col)[metric].idxmax()]
        
        # Plot individual equity curves of selected top strategies
        plt.figure(figsize=(7, 6))
        plt.subplot(2, 1, 1)
        for strategy in top_strategies.index:
            equity_curve = returns[strategy].copy()
            equity_curve = equity_curve.cumsum()
            plt.plot(equity_curve, label=strategy)
        
        plt.title(f'Invdividual Equity Curves of Selected Top Strategies based on {metric}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend(loc='upper left')
        
        # Calculate and plot the combined equity curve of the top strategies
        combined_equity_curve = returns[top_strategies.index].sum(axis=1)
        combined_equity_curve = combined_equity_curve.cumsum()
        
        plt.subplot(2, 1, 2)
        plt.plot(combined_equity_curve, label='Combined Equity Curve', color='blue')
        
        plt.title(f'Cumulative Performance of Top Strategies based on {metric}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print the correlation matrix for the selected strategies
        selected_returns = returns[top_strategies.index]
        correlation_matrix = selected_returns.corr()    
        # plot the correlation matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f'Correlation Matrix of Selected Top Strategies based on {metric}')
        plt.show()

        #calculating some statistics of the portfolios returns
        combined_pct_change = combined_equity_curve.pct_change().dropna()
        spy_pct_change = etf_data['SPY']
        correlation_to_spy = combined_pct_change.corr(spy_pct_change)
        print(f"Correlation of Portfolio to SPY based on {metric}: {correlation_to_spy:.4f}")
        
        # calculate beta
        covariance = combined_pct_change.cov(spy_pct_change)
        variance_spy = spy_pct_change.var()
        beta = covariance / variance_spy
        print(f"Beta of Portfolio to SPY based on {metric}: {beta:.4f}")

    @staticmethod
    def compare_metrics(returns, metrics_df, cluster_col='Cluster', metrics=['NetProfit', 'PNL/DD', 'Sharpe', 'Sortino']):
        """
        Compares the combined equity curves of selected top strategies based on different metrics.

        Args:
            returns (pd.DataFrame): DataFrame of strategy returns.
            metrics_df (pd.DataFrame): DataFrame of metrics for each strategy.
            cluster_col (str): Column name for cluster labels. Default is 'Cluster'.
            metrics (list): List of metrics to use for selecting top strategies. Default is ['NetProfit', 'PNL/DD', 'Sharpe', 'Sortino'].

        Returns:
            dict: Dictionary containing the combined equity curves for each metric.
        """

        #printing available features to create a portfolio from
        columns = ', '.join(metrics_df.columns)
        print(f"Available Features to select: {columns}")

        combined_curves = {}
        
        for metric in metrics:
            # Ensure metric column is numeric and handle NaN values
            metrics_df[metric] = pd.to_numeric(metrics_df[metric], errors='coerce')
            metrics_df = metrics_df.dropna(subset=[metric, cluster_col])

            top_strategies = metrics_df.loc[metrics_df.groupby(cluster_col)[metric].idxmax()]
            combined_equity_curve = returns[top_strategies.index].sum(axis=1).cumsum()
            combined_curves[metric] = combined_equity_curve
        
        plt.figure(figsize=(7, 4))
        
        for metric, equity_curve in combined_curves.items():
            plt.plot(equity_curve, label=f'Combined Equity Curve ({metric})')
        
        plt.title('Comparison of Portfolios by Different Metrics')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend(loc='upper left')
        plt.show()

        return combined_curves


    @staticmethod
    def plot_cluster_performance(returns, cluster_labels):
        unique_clusters = cluster_labels['Cluster'].unique()
        avg_performance = {}

        for cluster in unique_clusters:
            cluster_strategies = cluster_labels[cluster_labels['Cluster'] == cluster].index
            cluster_returns = returns[cluster_strategies].mean(axis=1)
            avg_performance[cluster] = cluster_returns.cumsum()

        # Plot the average performance of each cluster
        plt.figure(figsize=(7, 4))
        for cluster, performance in avg_performance.items():
            plt.plot(performance, label=f'Cluster {cluster} Avg Performance')

        plt.title('Equity Curve of Avg Performance of Each Cluster')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend(loc='upper left')
        plt.show()

    @staticmethod
    def plot_cluster_correlation_matrix(returns, cluster_labels):
        unique_clusters = cluster_labels['Cluster'].unique()
        avg_returns = pd.DataFrame()

        for cluster in unique_clusters:
            cluster_strategies = cluster_labels[cluster_labels['Cluster'] == cluster].index
            cluster_avg_returns = returns[cluster_strategies].mean(axis=1)
            avg_returns[f'Cluster {cluster}'] = cluster_avg_returns

        # Calculate the correlation matrix
        correlation_matrix = avg_returns.corr()

        # Plot the correlation matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Corr Matrix of Avg Returns Between Clusters')
        plt.show()

