#data_loading module
from data_loading import DataLoader
from feature_engineering import FeatureEngineering
from clustering import Clustering
from portfolio import Portfolio


def main():
    ##PARAMETERS
    raw_data_path = 'data/rawMCdata.csv'                    #strategy file export from multicharts

    #symbols for feature generation, pick at least 2
    etf_symbols = ['SPY', 'QQQ', 'DIA', 'IWM',              #equities
                   'TLT', 'IEF',                            #fixed income
                   'HYG', 'LQD',                            #corporate bonds 
                   'GLD', 'USO', 'GDX', 'SLV',              #commodities
                   'XLF', 'XLK', 'XLE', 'XLV', 'XLY',       #sectors & industries
                   'EFA', 'EEM', 'VWO', 'EWJ',              #international and emerging markets
                   'VNQ']                                   #real estate

    start_date = '2006-01-01'               
    end_date = '2025-01-01'


    # 1) DATA PROCESSING: cleaning, organizing data 
    returns_data = DataLoader.load_and_process(raw_data_path, 
                                               start_date,
                                               end_date)
    etf_data = DataLoader.load_etf_returns(etf_symbols, 
                                           start_date, 
                                           end_date)

    ## 2) FEATURE ENGINEERING
    feature_data = FeatureEngineering.create_features(returns_data, etf_data)
    print(feature_data)

    # 2.1) DIMENSIONALITY REDUCTION (PCA)
    components_df, _, _ = FeatureEngineering.apply_pca(feature_data, 
                                                       variance_threshold=0.85)

    ## 2.2) CREATE CUSTOM METRIC
    weights = [1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8]
    metrics = [
        'NetProfit_last_5Y', 'Sortino_last_5Y', 'PNL/DD_last_5Y', 'Corr_SPY_last_5Y',
        'NetProfit_all', 'Sortino_all', 'PNL/DD_all', 'Corr_SPY_all'
    ]
    custom_metric_name = 'CustomMetric'
    feature_data = FeatureEngineering.create_custom_metric(feature_data, 
                                                           weights, 
                                                           metrics, 
                                                           custom_metric_name)
    print(f'feature DATA: \n {feature_data}')

    ## 2.3) FILTER AFTER PCA, BASED ON CUSTOM METRIC
    # EXAMPLE: SELECT TOP 50 STRATEGIES BY THE CUSTOM METRIC
    top_n_strats = 50
    # Sort from highest to lowest on that custom metric
    sorted_by_custom = feature_data[custom_metric_name].sort_values(ascending=False)
    top_strategies = sorted_by_custom.index[:top_n_strats]
    # Filter the PCA components to only those top 50 strategies
    filtered_components_df = components_df.loc[top_strategies]

    ## 3) CLUSTERING: K-Means ON FILTERED STRATEGIES
    #Clustering.elbow_method(filtered_components_df, end=20, n_init=150) #uncomment if you want elbow
    cluster_column = Clustering.cluster_data(filtered_components_df,
                                             n_clusters=6, 
                                             n_init=250)

    ## 4) PORTFOLIO CREATION: SELECT TOP STRATEGIES FROM EACH CLUSTER
    # We only have cluster labels for the top 50, so join on them
    grouped_data = feature_data.join(cluster_column, how='inner')  # keep only top 50

    Portfolio.compare_metrics(returns_data, 
                              grouped_data,
                              metrics=['NetProfit_last_4Y', 'PNL/DD_last_4Y','CustomMetric', 'Corr_SPY_last_4Y'])
    
    Portfolio.plot_cluster_performance(returns_data, grouped_data)
    Portfolio.plot_cluster_correlation_matrix(returns_data, grouped_data)

    #CREATE SAMPLE PORTFOLIOS
    Portfolio.create(returns_data, grouped_data, etf_data, metric='CustomMetric')
    Portfolio.create(returns_data, grouped_data, etf_data, metric='NetProfit_last_3Y')

    pass

if __name__ == '__main__':
    main()
