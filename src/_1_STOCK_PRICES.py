from _0_helper_functions import *

'''
                                            << STOCK_PRICES (module) >>
    => Class Initialization Inputs:
        (1) nan_threshold                 : float | Threshold for NaN values, above which tickers will be dropped.
        (2) delete_tickers_w_missing_rows : bool | Flag to determine if tickers with missing rows should be dropped.

    => Available Methods:
        (1) get_data_from_yf(tickers, start_date, end_date):
            => Retrieves stock price data from Yahoo Finance for given tickers within the specified date range.

            => Inputs: 
                tickers     : list  | List of stock ticker symbols.
                start_date  : str   | Start date in 'YYYY-MM-DD' format.
                end_date    : str   | End date in 'YYYY-MM-DD' format.

            => Outputs: 
                DataFrame   | Data frame with stock prices for the specified tickers and date range.
        
        (2) get_clean_stock_data(tickers_list=[], path_to_prices_df=None, yf_start=None, yf_end=None):
            => Retrieves and cleans stock price data using either a CSV file or Yahoo Finance.
                * if csv file is passed it should have columns titled as such:
                    ticker   |      date    | open | high | low | close | volume 
                        A      '2023-01-01' ...
                        A      '2023-01-02' ...
                        ⋮
                        B      '2023-01-01' ... 
                        B      '2023-01-02'

            => Inputs:
                tickers_list         : list | List of stock ticker symbols (optional).
                path_to_prices_df    : str  | Path to CSV file with stock prices (optional).
                yf_start             : str  | Start date for Yahoo Finance data retrieval (optional).
                yf_end               : str  | End date for Yahoo Finance data retrieval (optional).
                start_date           : str  | start_date to get data in a range (optional)
                end_date             : str  | end_date to get data in a range (optional)

            => Outputs:
                DataFrame            | Cleaned data frame with stock prices for the specified tickers.
'''

class STOCK_PRICES():
    def __init__(self, nan_threshold : float = 0.0, delete_tickers_w_missing_rows : bool = True):
        # (1) Initialize parameters to conduct clean
        self.delete_tickers_w_missing_rows : bool  = delete_tickers_w_missing_rows
        self.nan_threshold                 : float = nan_threshold
    
    '''
        get_data_from_yf: returns a data frame with stock prices for tickers in 'tickers' list from
        'start_date' to 'end_date'. Data frame format:
                Ticker   |      Date    | Open | High | Low | Close | Volume 
                A          '2023-01-01' ...
                A          '2023-01-02' ...
                ⋮
                B          '2023-01-01' ... 
                B          '2023-01-02'
    '''
    def get_data_from_yf(self, tickers : list, start_date : str, end_date : str):
        ticker_data_frames = []
        
        for ticker in tickers:
            data = yf.download(ticker, start = start_date, end = end_date, interval = '1d')
            data['Ticker'] = ticker
            data = data.reset_index()
            ticker_data_frames.append(data)
        
        combined_data = pd.concat(ticker_data_frames, ignore_index=True)
            
        return combined_data
    
    '''
        get_clean_stock_data: uses class init parameters to retrieve clean stock data.
            Input:
                => 'path_to_prices_df' : str - If left empty (None), the function will retrieve data from 
                    Yahoo Finance. If provided, it will open the data from the specified CSV file and return 
                    a cleaned version. The CSV file should contain data in a dataframe with the following 
                    column titles: 
                            ticker   |      date    | open | high | low | close | Volume 
                            A          '2023-01-01' ...
                            A          '2023-01-02' ...
                            ⋮
                            B          '2023-01-01' ... 
                            B          '2023-01-02'
    ''' 
    def get_clean_stock_data(self, tickers_list: list = [], path_to_prices_df : str = None, yf_start = None, yf_end = None,
                             start_date : str = None, end_date : str = None):
        # (1) Open price csv or use Yahoo Finance to get data frame 
        prices_df = pd.DataFrame()
        if path_to_prices_df == None:
            prices_df = self.get_data_from_yf(tickers = tickers_list, start_date = yf_start, end_date = yf_end)
        else:
            prices_df = pd.read_csv(path_to_prices_df)
            # if user did not pass in a list, try to clean data from all available tickers in df 
            if tickers_list == []: 
                tickers_list = getTickers(prices_df)

        prices_df.rename(columns = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Date": "date", "Ticker": "ticker"}, 
                         inplace = True)
        
        # if user passed in day range: 
        if (start_date != None and end_date != None):
            prices_df = get_data_in_date_range(df = prices_df, start_date = start_date, end_date = end_date)
                
        # (2) Drop columns that are not: open , high, low, close, volume, datadate
        for col in prices_df:
           if col not in ["ticker", "open", "high", "low", "close", "date", "volume", "VWAP"]:
              prices_df.drop(col, axis = 1, inplace = True)
    
        # (3) Drop the all tickers that are not in 'tickers_list'
        prices_df = dropTickers(df = prices_df, tickersToKeep = tickers_list)

        # (4) Drop tickers whose NaN Value % is above threshold
        high_nan_tickers : list = emptyTickers(df = prices_df, symbols = tickers_list, threshold = self.nan_threshold)
        ok_tickers = []
        for ticker in tickers_list:
            if ticker not in high_nan_tickers:
                ok_tickers.append(ticker)
        prices_df = dropTickers(df = prices_df, tickersToKeep = ok_tickers)

        # (5) Only keep tickers that have same number of rows (if specified by class attribute)
        if self.delete_tickers_w_missing_rows == True:
            results = rows_per_ticker(df = prices_df, symbols = getTickers(prices_df))
            most_common_number_of_rows = results[1]
            
            # Remove tickers that have more than/less rows than that the most popular number of rows
            tickers_w_common= [] # List with tickers whose #rows == most popular # of rows
            for ticker in getTickers(prices_df):
                (rows, _) = getSymbolData(df = prices_df, symbol = ticker).shape
                if rows == most_common_number_of_rows:
                    tickers_w_common.append(ticker)
            prices_df = dropTickers(prices_df, tickers_w_common)
        
        # (6) Return data frame with column order: | date | ticker | open | high | low | close | volume 
        final_prices_df = pd.DataFrame()
        for column in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']:
            final_prices_df[column] = prices_df[column].tolist()
        
        return final_prices_df
