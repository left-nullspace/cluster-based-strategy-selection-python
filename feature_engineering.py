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
        
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning) #supress warnings

        print('Creating features...')
        features_df = pd.DataFrame(index=returns.columns)
        time_windows = {'': None, '_3m': '3M', '_6m': '6M', '_1y': '1Y', '_2y': '2Y', '_3y': '3Y', '_4y': '4Y', '_5y': '5Y', '_10y': '10Y'}
        for suffix, window in time_windows.items():
            if window:
                window_returns = returns.last(window)
                window_etf_returns = etf_returns.last(window)
            else:
                window_returns = returns
                window_etf_returns = etf_returns
            features_df = FeatureEngineering.add_features(window_returns, window_etf_returns, features_df, suffix)
        return features_df

    @staticmethod
    def add_features(returns, etf_returns, features_df, suffix=''):
        # Calculate features
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
        print('=== PRINCIPAL COMPONENT ANALYSIS ===')

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
                                                    

