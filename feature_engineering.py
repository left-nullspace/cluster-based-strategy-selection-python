from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings

class FeatureEngineering:
    @staticmethod
    def calculate_max_drawdown(cumulative_profit_series):
        roll_max = cumulative_profit_series.cummax()
        drawdown = roll_max - cumulative_profit_series
        max_drawdown = drawdown.max()
        return max_drawdown

    @staticmethod
    def create_features(returns, etf_returns):
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)  # suppress warnings

        print('creating features... please wait')
        features_df = pd.DataFrame(index=returns.columns)
        time_windows = ['1Y', '2Y', '3Y', '4Y', '5Y', '8Y', '10Y']
        
        # Calculate features for the full dataset
        features_df = FeatureEngineering.add_features(returns, etf_returns, features_df, '_all')
        
        # Calculate features for 'last' time windows
        for window in time_windows:
            window_returns_last = returns.last(window)
            window_etf_returns_last = etf_returns.last(window)
            suffix_last = f'_last_{window}'
            features_df = FeatureEngineering.add_features(window_returns_last, window_etf_returns_last, features_df, suffix_last)
        
        # Calculate features for 'first' time windows
        for window in time_windows:
            window_returns_first = returns.first(window)
            window_etf_returns_first = etf_returns.first(window)
            suffix_first = f'_first_{window}'
            features_df = FeatureEngineering.add_features(window_returns_first, window_etf_returns_first, features_df, suffix_first)
        
        return features_df


    @staticmethod
    def add_features(returns, etf_returns, features_df, suffix=''):
        # calculate features
        cumulative_profit = returns.cumsum()
        total_profit = returns.sum()
        max_drawdown = cumulative_profit.apply(FeatureEngineering.calculate_max_drawdown)
        std_dev = returns.std()
        mean_return = returns.mean()
        upside_std = returns[returns > 0].std()
        downside_std = returns[returns < 0].std()
        sharpe_ratio = (mean_return / std_dev.replace(0, np.nan)) * np.sqrt(252)
        sortino_ratio = (mean_return / downside_std.replace(0, np.nan)) * np.sqrt(252)
        active_days = returns.astype(bool).sum(axis=0)
        total_days = len(returns)

        # Prepare to add new data
        new_data = {
            f'NetProfit{suffix}': total_profit,
            f'MaxDD{suffix}': max_drawdown,
            f'PNL/DD{suffix}': total_profit / max_drawdown.replace(0, np.nan),
            f'StdDev{suffix}': std_dev,
            f'Sharpe{suffix}': sharpe_ratio,
            f'Sortino{suffix}': sortino_ratio,
            f'UpStdDev{suffix}': upside_std,
            f'DownStdDev{suffix}': downside_std,
            f'Up/DownStdDev{suffix}': upside_std / downside_std,
            f'PercentInMkt{suffix}': active_days / total_days
        }

        # Concatenate new data
        new_data_df = pd.DataFrame(new_data, index=returns.columns)
        features_df = pd.concat([features_df, new_data_df], axis=1)

        # Calculate and add correlations
        for column in returns.columns:
            filtered_series = returns[column][returns[column] != 0]
            for etf_column in etf_returns.columns:
                combined_df = pd.concat([filtered_series, etf_returns[etf_column]], axis=1).dropna()
                correlation = combined_df.corr().iloc[0, 1]
                features_df.at[column, f'Corr_{etf_column}{suffix}'] = correlation

        return features_df
    
    @staticmethod
    def apply_pca(feature_data, variance_threshold = 0.85, svd_solver='full', whiten=True):

        #standardize
        print(f"number of features BEFORE dropping NaN columns: {feature_data.shape[1]}")
        feature_data = feature_data.dropna(axis=1) #drop columns with NaN
        print(f"number of features AFTER dropping NaN columns: {feature_data.shape[1]}")
        scaled_data = StandardScaler().fit_transform(feature_data) #z-normalization

        #create PCA object
        pca = PCA(svd_solver=svd_solver, whiten=whiten)
        pca.fit(scaled_data)

        # calculate cumulative explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        print(f"Cumulative variance explained by components:\n {cumulative_variance}")
        
        # determine the number of components to retain based on the variance threshold
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Number of components to retain (variance threshold of {variance_threshold*100}%): {n_components}")
        
        # Recreate PCA with the determined number of components
        pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten)
        principal_components = pca.fit_transform(scaled_data)
        
        # create a DataFrame from the principal components
        components_df = pd.DataFrame(data=principal_components,
                                    columns=[f'PC{i+1}' for i in range(n_components)],
                                    index=feature_data.index)  # Preserving the original index
        print(components_df.head())
        
        return components_df, explained_variance, cumulative_variance
    
    @staticmethod
    def create_custom_metric(features_df, weights, metrics, metric_name):
        """
        Creates a custom metric by normalizing specified metrics including handling
        correlation metrics differently, then computing a weighted sum of these metrics.

        Parameters:
            features_df (DataFrame): DataFrame containing the features.
            weights (list): List of weights for each metric.
            metrics (list): List of metric column names to include in the custom metric.
            metric_name (str): Name for the new custom metric column.

        Returns:
            DataFrame: Updated DataFrame with the new custom metric.
        """
        # Initialize an empty DataFrame for normalized metrics
        normalized_metrics = pd.DataFrame(index=features_df.index)
        
        # Normalize specified metrics using z-score, handle correlations differently
        for metric in metrics:
            if 'Corr' in metric:
                # Transform correlations: 1 - |Correlation|
                transformed = 1 - np.abs(features_df[metric])
                # Normalize transformed data
                normalized_metrics[metric] = (transformed - transformed.mean()) / transformed.std()
            else:
                # Regular z-score normalization for other metrics
                normalized_metrics[metric] = (features_df[metric] - features_df[metric].mean()) / features_df[metric].std()

        # Compute weighted sum of the normalized metrics
        features_df[metric_name] = np.dot(normalized_metrics, weights)

        return features_df

