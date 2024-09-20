from _0_helper_functions import *
'''
                                    << DATA_MANIPULATOR (module) >>
    => Class Initialization Inputs:
        (1) prices_path : string | path to file with OHLCV data of stocks

    => Available Methods:
        (1) initialize_data():
            => Loads the stock data from the file specified by "prices_path" and creates a 
               class attribute "prices_df" containing this data.

            => Inputs: None
            
            => Outputs: None

        (2) get_data_between_dates(symbol, start_date, end_date):
            => Retrieves price data for the specified stock symbol within the given date range.

            => Inputs: 
                symbol     : string | stock ticker symbol
                start_date : string | start date in 'YYYY-MM-DD' format
                end_date   : string | end date in 'YYYY-MM-DD' format

            => Outputs: 
                DataFrame | price data for the specified symbol between the start and end dates

        (3) calculate_hedge_ratio(close_prices_1, close_prices_2):
            => Calculates the hedge ratio, which is the proportion of the second asset needed 
               to hedge one unit of the first asset, based on their closing price series.

            => Inputs:
                close_prices_1 : Series or array | closing prices of the first asset
                close_prices_2 : Series or array | closing prices of the second asset

            => Outputs: 
                float | the hedge ratio
        
        (4) calculate_spread(close_prices_1, close_prices_2, hedge_ratio):
            => Calculates the spread between two assets' closing prices

            => Inputs:
                close_prices_1 : Series or array | closing prices of the first asset
                close_prices_2 : Series or array | closing prices of the second asset
                hedge_ratio    : float           | the hedge ratio
        
            => Outputs:
                Series | the calculated spread
        
        (5) calculate_cointegration(close_prices_1, close_prices_2):
            => Determines whether two time series are cointegrated, indicating a long-term 
               equilibrium relationship between them.

            => Performs the Engle-Granger two-step cointegration test on the closing prices of
               two assets. It extracts the test statistic, p-value, and critical value to make
               the decision.

            => Inputs:
                close_prices_1 : Series or array | closing prices of the first asset
                close_prices_2 : Series or array | closing prices of the second asset

            => Outputs:
                bool | True if the series are cointegrated, False otherwise
        
        (6) calculate_zcore(spread, pWindow):
            => Calculates how many stardard deviations is the spread away from its mean (Z-score) 
               for each day of a given spread series using a rolling window.

            => Inputs:
                spread   : Series or array | the spread between two time series
                pWindow  : int             | the rolling window size for calculating mean and std deviation
            
            => Outputs:
                list | Z-score values for the spread series over the rolling window
'''

class DATA_MANIPULATOR():
    def __init__(self, prices_path):
        # (1) Intialize parameters
        self.prices_path = prices_path

        # (2) Flag - keeps track if data has been initialized
        self.is_init = False

        # (3) Initialize parameter - pandas df of price data 
        self.prices_df = pd.DataFrame()

    '''  
    initialize_data: reads pandas df from prices_path passed into the class 
        => input  : None
        => output : None 
    '''
    def initialize_data(self):
        # (1) Read Prices Data
        self.prices_df = pd.read_csv(self.prices_path)

        # (ii) Rename Columns of Data to Open, High, Low, Close, Volume
        self.prices_df.rename(columns = {   "open"        : "Open" ,
                                            "high"        : "High",
                                            "low"         : "Low",
                                            "close"       : "Close",
                                            "volume"      : "Volume",
                                            "datadate"    : "Date",
                                            "date"        : "Date",
                                            "OPEN_PRC"    : "Open",
                                            "HIGH_1"      : "High",
                                            "LOW_1"       : "Low",
                                            "Close Price" : "Close",
                                            "Ticker"      : "ticker"},
                                inplace = True)

        # (2) Drop Unnamed Column
        if "Unnamed: 0" in self.prices_df.columns:
            self.prices_df.drop("Unnamed: 0", axis = 1, inplace = True)

        # (3) Make Date the Index
        self.prices_df.set_index("Date", inplace = True)

        # (4) Convert Date to Datetime
        self.prices_df.index = pd.to_datetime(self.prices_df.index)

        # (5) Mark The Data As Initialized
        self.is_init = True
        return None 

    '''
        is_initialized : returns true if pandas df has been created from 
        the path to price data passed into class 
    '''
    def is_initialized(self):
        return self.is_init

    ''' 
        get_data_between_dates : returns a dataframe of data from a specific
        symbol between two days 
        => Inputs:
            (1) symbol     : str     "AAPL"
            (2) start-date : str     "YYYY-MM-DD"
            (3) end-date   : str     "YYYY-MM-DD"
    '''
    def get_data_between_dates(self, symbol : str, start_date : str, end_date : str):
        # (i) Get Data Frame With Ony Symbol
        ticker_df = getSymbolData(df = self.prices_df, symbol = symbol)

        # (ii) Filter Symbol's Data By Date
        filtered = ticker_df.loc[start_date : end_date]
        return filtered

    '''
        calculate_hedge_ratio: calculates hedge ratio between the closing prices of 2 assets 
        by performing an Ordinary Least Squares (OLS) regression where 'close_prices_1' is 
        the dependent variable and 'close_price_2' is the independent variable. 

        => Terminology:
            'hedge ratio': number of units of the second asset needed to hedge one unit of 
            the first asset. 

        => Inputs:
            (1) close_prices_1 : series / list
            (2) close_prices_2 : series / list

        => Outputs: 
            hedge ratio (𝛽 : float). 

    '''
    def calculate_hedge_ratio(self, close_prices_1, close_prices_2):
        model = sm.OLS(close_prices_1, close_prices_2).fit()
        hedge_ratio = model.params[0]
        return hedge_ratio

    '''
        calculate_spread: calculates the daily spread of two assets using their closing 
        prices and their hedge_ratio 
        
        => Terminology:
            'spread': the difference between the adjusted prices of the two assets 

        => Inputs: 
            (1) close_prices_1 : list 
            (2) close_prices_2 : list 
            (3) hedge_ratio    : float 
        
        => Outputs:
            spread (series)
    '''
    def calculate_spread(self, close_prices_1, close_prices_2, hedge_ratio):
        spread = pd.Series(close_prices_1) - (hedge_ratio * pd.Series(close_prices_2))
        return spread

    '''
        calculate_cointegration: returns true if the closing prices of asset 1 are 
        cointegrated to the closing prices of asset 2 based on the Engle-Granger two-step
        cointegration test (uses the 5% significance level)

        => Terminology: 
            cointegration: when two time series have a long-term equilibrium relationship}
        
        => Inputs: 
            (1) close_prices_1 : list 
            (2) close_prices_2 : list

        => Outputs:
            cointegration flag : boolean 

    '''
    def calculate_cointegration(self, close_prices_1, close_prices_2):
        coint_res = coint(close_prices_1, close_prices_2)
        t_value = coint_res[0]
        p_value = coint_res[1]
        c_value = coint_res[2][1]
        coint_flag = True if p_value < 0.5 and t_value < c_value else False
        return coint_flag

    '''
        calculate zscore: returns list of zscore values for the spread using a 'pWindow' 
        day to calculate zscore 
        => Terminology:
            Z-score: how many standard deviations a data point is from the mean of a dataset 

        => Inputs:
            (1) spread  : float list 
            (2) pWindow : integer 
        
        => Output: 
            Z-score : list of floats 
    '''
    def calculate_zcore(self, spread, pWindow):
        # Ensure spread is a Pandas Series
        spread_series = pd.Series(spread)

        # Calculate rolling mean and standard deviation
        rolling_mean = spread_series.rolling(window=pWindow).mean()
        rolling_std  = spread_series.rolling(window=pWindow).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-10)

        # Calculate Z-score
        z_score = (spread_series - rolling_mean) / rolling_std

        # Handle initial NaN values
        z_score = z_score.fillna(0)

        z_score_list = z_score.values.tolist()
        return z_score_list

    def calculate_rolling_cointegration(self, close_prices_1, close_prices_2, window):
        # (i) Initialize Output Variable
        coint_list = []

        # (ii) Populate First "Window - 1" Elements of List with np.nan
        coint_list = coint_list + ([np.nan] * window)

        # (iii) Calculate Rolling cointegration flags
        for i in range(window, len(close_prices_1)):
            coint_res  = self.calculate_cointegration(close_prices_1[i-window:i], close_prices_2[i-window:i])
            coint_list.append(int(coint_res))

        return coint_list
