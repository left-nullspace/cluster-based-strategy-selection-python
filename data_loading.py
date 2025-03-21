import yfinance as yf
import pandas as pd

#import portfolio data from multicharts portfoliotrader: list of trades, SEMICOLON separated, as a csv file
pd.options.mode.chained_assignment = None
class DataLoader:

    @staticmethod
    def load_and_process(file_path, start_date, end_date):
        #read in raw multicharts data
        data = pd.read_csv(file_path, sep=';') #separate w semicolon
        print(f"data loaded from {file_path}, with columns: {data.columns.tolist()}")

        #process data
        extracted_data = data.iloc[:, [1, 8, 10]] #extract relevant columns
        extracted_data['exitTime'] = pd.to_datetime(extracted_data['exitTime']) #parse exittime to datetime
        extracted_data['symbol'] = extracted_data['symbol'].apply(lambda x: x.split('<br>')[0])

        #filter data between specified start and end dates
        mask = (extracted_data['exitTime'] >= start_date) & (extracted_data['exitTime'] <= end_date)
        filtered_data = extracted_data.loc[mask]
        
        processed_data = filtered_data.pivot(index='exitTime', 
                                              columns='symbol', 
                                              values='profit')
        processed_data = processed_data.fillna(0)
        print(f"PROCESSED STRATEGY DATA: \n {processed_data.head(3)}")
        return processed_data #returns
    
    @staticmethod
    def load_etf_returns(etf_symbols, start_date, end_date):
        # Download data with auto_adjust and rounding enabled.
        etf_data = yf.download(
            etf_symbols,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            rounding=True
        )
        # With auto_adjust=True, use the adjusted 'Close' column to compute returns.
        etf_data = etf_data['Close'].pct_change().dropna()   # raw percentage returns

        print(f"Data loaded from YAHOOFINANCE for metric and feature calculations: \n{etf_data.head(3)}")
        return etf_data