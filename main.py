#data_loading module
from data_loading import DataLoader
from feature_engineering import FeatureEngineering
from clustering import Clustering
from portfolio import Portfolio


def main():
    ##PARAMETERS
    raw_data_path = 'data/rawMCdata.csv'                    #strategy file export from multicharts

    #symbols for feature generation
    etf_symbols = ['SPY', 'QQQ', 'DIA', 'IWM',              #equities
                   'TLT', 'IEF',                            #fixed income
                   'HYG', 'LQD',                            #corporate bonds 
                   'GLD', 'USO', 'GDX', 'SLV',              #commodities
                   'XLF', 'XLK', 'XLE', 'XLV', 'XLY',       #sectors & industries
                   'EFA', 'EEM', 'VWO', 'EWJ',              #international and emerging markets
                   'VNQ']                                   #real estate
    #etf symbols for feature generation, pick at least 2
    start_date = '2007-01-01'               
    end_date = '2024-06-01'



    # 1)DATA PROCESSING: cleaning, organizing data -----

    # load and process data
    returns_data = DataLoader.load_and_process(raw_data_path, 
                                               start_date,
                                                 end_date)
    # load the ETF data for creating features
    etf_data = DataLoader.load_etf_returns(etf_symbols, 
                                           start_date, 
                                           end_date)
    #split data (not for this problem, solution is unknown)



    ## 2) FEATURE ENGINEERING: Creation -----
    feature_data = FeatureEngineering.create_features(returns_data,
                                                      etf_data)
    print(feature_data)
    # dimensionality Reduction with PCA
    components_df, _, _ = FeatureEngineering.apply_pca(feature_data, 
                                                       variance_threshold=0.85)

    ## 3) CLUSTERING: using principal components to cluster the strategies
    #Clustering.elbow_method(components_df, end=20, n_init=150) #uncomment if want to look at elbow plot
    cluster_column = Clustering.cluster_data(components_df,
                                             n_clusters=8, 
                                             n_init=250)
    grouped_data = feature_data.join(cluster_column) #now features with cluster col



    ## 4) PORTFOLIO CREATION: Select top strategies from each cluster based on a metric & compare equity curves
    
    #analyzing clusters and a few portfolios
    Portfolio.compare_metrics(returns_data, 
                              grouped_data,
                              metrics=['NetProfit', 'PNL/DD', 'Sharpe', 'Sortino_4y'])     #compare creating diff portfolios
    Portfolio.plot_cluster_performance(returns_data, grouped_data)                      #equity curve of avg of each cluster
    Portfolio.plot_cluster_correlation_matrix(returns_data, grouped_data)               #correlation btwn avg of each cluster
    
    #design own portfolio for further analysis
    Portfolio.create(returns_data, grouped_data, metric='NetProfit_4y')




    #Evaluation
    pass

if __name__ == '__main__':
    main()
