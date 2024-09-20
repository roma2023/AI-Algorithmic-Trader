from _0_helper_functions import *
from _5_DATA_MANIPULATOR import DATA_MANIPULATOR
from _6_ZSCORE_STRATEGY  import ZSCORE_STRATEGY

'''
                                    << PAIRS_BACKTESTER (module) >>
    => Class Initialization Inputs:
        (1) strategy_obj     : ZSCORE_STRATEGY | object of ZSCORE_STRATEGY class
        (2) initial_capital  : int             | initial capital for the backtest
        (3) fixed_allocation : float           | fixed allocation for each trade
        (4) risk_per_trade   : float           | risk percentage per trade

    ------------ User-facing Methods --------------
        (1) backtest(show_output = False):
            => Executes the full backtest pipeline, optionally prints results, and returns the results dictionary.
            => Inputs:
                show_output : bool | whether to print the results
            => Outputs:
                dict | results dictionary with all the backtest metrics

    ------------ Backend Methods --------------
        (1) add_returns_to_trade_df(long_symbol, long_entry_price, short_symbol, short_entry_price, trade_df):
            => Adds accumulated returns for long and short positions and net returns to the trade DataFrame.
            => Inputs:
                long_symbol       : string    | ticker symbol of the long position
                long_entry_price  : float     | entry price for the long position
                short_symbol      : string    | ticker symbol of the short position
                short_entry_price : float     | entry price for the short position
                trade_df          : DataFrame | DataFrame containing trade data
            => Outputs:
                DataFrame | updated trade DataFrame with returns columns

        (2) calcuate_new_portfolio_value(curr_port_value, risk_per_trade, fixed_allocation, long_returns, short_returns):
            => Calculates the new portfolio value based on the returns from long and short positions.
            => Inputs:
                curr_port_value  : float | current portfolio value
                risk_per_trade   : float | risk percentage per trade
                fixed_allocation : float | fixed allocation per trade
                long_returns     : float | returns from the long position
                short_returns    : float | returns from the short position
            => Outputs:
                float | updated portfolio value

        (3) build_report_df():
            => Constructs the 'report_df' DataFrame with details of each trade and updates the portfolio value.
            => Outputs:
                None

        (4) get_sharpe_ratio(years, risk_free_rate = 0.005):
            => Calculates the Sharpe ratio based on the trade returns in 'report_df'.
            => Inputs:
                years           : float | number of years in the backtest period
                risk_free_rate  : float | risk-free rate for Sharpe ratio calculation
            => Outputs:
                float | Sharpe ratio

        (5) build_results_dict():
            => Summarizes the results of all trades in 'report_df' and stores them in 'results_dict'.
            => Outputs:
                None

        (7) print_results():
            => Prints the results of the backtest stored in 'results_dict'.
            => Outputs:
                None
'''

class PAIRS_BACKTESTER():
    def __init__(self, strategy_obj : ZSCORE_STRATEGY, initial_capital : int , fixed_allocation : float , risk_per_trade : float):
        # (1) Initialize strategy object & parametrs
        self.strategy_obj      = strategy_obj
        self.fixed_allocation  = fixed_allocation
        self.risk_per_trade    = risk_per_trade
        self.initial_capital   = initial_capital

        # (2) Initialize report_df - holds information about each trade performed by the backtest 
        self.report_df = pd.DataFrame(columns = ["Start", "End", "Duration", "Long Symbol", "Short Symbol", "Entry Long", "Exit Long", 
                                                 "Entry Short", "Exit Short", "Return on Long", "Return on Short", "Net Return", 
                                                 "Portfolio Value"])
        
        # (3) Initilize two arrays to store the dataframes of winning and loosing trades
        self.winning_trades = []
        self.loosing_trades = []

        # (4) Initialize portafolio value to initial capital 
        self.portfolio_value = initial_capital 

        # (5) Initialize results_dict - holds summary of all trades in 'report_df'
        self.results_dict = dict()

    '''
        add_returns_to_trade_df: Adds three columns to a trade_df:
            1. long_returns - Accumulated returns from the long position (up to each day).
            2. short_returns - Accumulated returns from the short position (up to each day).
            3. net_returns - Daily sum of long_returns and short_returns.
        => Helper for 'build_report_df' function  

    '''
    def add_returns_to_trade_df(self, long_symbol, long_entry_price, short_symbol, short_entry_price, trade_df):
        long_returns  = []  
        short_returns = [] 
        net_returns   = []
        for index, row in trade_df.iterrows():
            curr_long_price     = row[f"Close_{long_symbol}"]
            curr_return_on_long = ((curr_long_price - long_entry_price) / long_entry_price) * 100

            curr_short_price = row[f"Close_{short_symbol}"]
            curr_return_on_short = ((short_entry_price - curr_short_price) / curr_short_price) * 100

            curr_net_return = (curr_return_on_long + curr_return_on_short) 

            long_returns.append(curr_return_on_long)
            short_returns.append(curr_return_on_short)
            net_returns.append(curr_net_return)
        # Remove 1st element from each list since we are not actually in the trade then
        trade_df['Long Return']  = [np.nan] + long_returns[1:]
        trade_df['Short Return'] = [np.nan] + short_returns[1:]
        trade_df['Net Return']   = [np.nan] + net_returns[1:]

        return trade_df
    
    '''
        long and short positions, current porfolio value and returns updated 
        portfolio value 
        => Helper for 'build_report_df' function  
    '''
    def calcuate_new_portfolio_value(self, curr_port_value, risk_per_trade, fixed_allocation, 
                                     long_returns, short_returns):
        # (1) Calculate position size & long/short investments 
        position_size    = curr_port_value * risk_per_trade
        long_investment  = position_size * fixed_allocation
        short_investment = position_size * fixed_allocation

        # (2) Calculate profits for both investments & update portfolio value 
        long_profit  = long_investment  * (long_returns  / 100) 
        short_profit = short_investment * (short_returns / 100)
        net_profit   = long_profit + short_profit
        
        new_portfolio_value = curr_port_value + net_profit
        return new_portfolio_value

    '''
        build_report_df: builds up 'report_df' class attribute by gathering start, end, duration, 
        and returns of each trade in sequential time order. In addition, it updates the class attribute 
        'portfolio_value' with the profits each trade has.
    '''
    def build_report_df(self):
        # (1) Retrieve 'backtesting df' from strategy object 
        ''' | Open_Symbol_1 | Close_Symbol_1 | Open_Symbol_2 | Close_Symbol_2 | XXX | YYY... | Signal |'''
        backtest_df = self.strategy_obj.build_backtester_df() 

        # (2) Drop rows with ("NONE", "NONE")
        backtest_df = backtest_df[backtest_df['Signal'] != ('NONE', 'NONE')]

        # (3) Split backtest_df into single trade data frames, i.e split at each ("CLOSE", "CLOSE")
        backtest_df.reset_index(inplace = True)                                                  # (i)   Add index column to backtest_df
        split_indices = backtest_df.index[backtest_df['Signal'] == ('CLOSE', 'CLOSE')].tolist()  # (ii)  Get indices of all ("CLOSE", "CLOSE")
        split_indices = [i + 1 for i in split_indices]                                           # (iii) Add one to all index to keep in the CLOSE, CLOSE in each trade df
        trades_dataframes = np.split(backtest_df, split_indices)                                 # (iv)  Split backtest_df at all indices of ("CLOSE", "CLOSE")  

        # (4) Populate 'report_df' class attribute with the information of each trade in 'trades_dataframes'
        for trade_df in trades_dataframes:
            if trade_df.empty or len(trade_df) < 2:
                continue 

            # Build dictionary with trade's information
            trade_signal = trade_df['Signal'].iloc[0]
            trade_dict = dict()
            trade_dict['Start']        = trade_df['Date'].iloc[1] # (i) we use 1 and not 0 bcs we use open price of next day to begin trade 
            trade_dict['End']          = trade_df['Date'].iloc[-1]
            trade_dict['Duration']     = (trade_dict['End'] - trade_dict['Start']).days
            trade_dict['Long Symbol']  = self.strategy_obj.Symbol_1 if trade_signal == ("BUY", "SELL") else self.strategy_obj.Symbol_2
            trade_dict['Short Symbol'] = self.strategy_obj.Symbol_1 if trade_signal == ("SELL", "BUY") else self.strategy_obj.Symbol_2
            trade_dict['Entry Long']   = trade_df[f'Open_{trade_dict['Long Symbol']}'].iloc[1]
            trade_dict['Exit Long']    = trade_df[f'Close_{trade_dict['Long Symbol']}'].iloc[-1]
            trade_dict['Entry Short']  = trade_df[f'Open_{trade_dict['Short Symbol']}'].iloc[1]
            trade_dict['Exit Short']   = trade_df[f'Close_{trade_dict['Short Symbol']}'].iloc[-1]
            
            # Augment trade_df with return columns by calling class helper function 
            trade_df = self.add_returns_to_trade_df(long_symbol       = trade_dict['Long Symbol'], 
                                                    long_entry_price  = trade_dict['Entry Long'], 
                                                    short_symbol      = trade_dict['Short Symbol'], 
                                                    short_entry_price = trade_dict['Entry Short'], 
                                                    trade_df = trade_df)
            
            trade_dict['Return on Long']  = trade_df['Long Return'].iloc[-1]
            trade_dict['Return on Short'] = trade_df['Short Return'].iloc[-1] 
            trade_dict['Net Return']      = trade_df['Net Return'].iloc[-1] 

            # Append trade_df to W/L trades list (which is a class attribute) depending on the trade's net return
            if trade_dict['Net Return'] > 0:
                self.winning_trades.append(trade_df)
            else:
                self.loosing_trades.append(trade_df)
            
            # Update the portafolio's value using class helper method
            self.portfolio_value = self.calcuate_new_portfolio_value(curr_port_value  = self.portfolio_value,
                                                                     risk_per_trade   = self.risk_per_trade,
                                                                     fixed_allocation = self.fixed_allocation,
                                                                     long_returns     = trade_dict['Return on Long'],
                                                                     short_returns    = trade_dict['Return on Short'])
            trade_dict['Portfolio Value'] = self.portfolio_value
            
            # Make 'trade_dict' a new row in class attribute 'report_df'
            new_row = pd.DataFrame([trade_dict])
            self.report_df = pd.concat([self.report_df, new_row], ignore_index = True, sort = False)

        return
    
    '''
        get_sharpe_ratio: calculates the sharpe ratio using report_df
    '''
    def get_sharpe_ratio(self, years, risk_free_rate = 0.005):
        df = self.report_df.copy()
        net_returns = np.array(df['Net Return'].tolist())

        mean_return = np.mean(net_returns)
        std_dev_return = np.std(net_returns)

        trades_per_year = len(net_returns) / years

        annualized_mean_return = mean_return * trades_per_year
        annualized_std_dev_return = std_dev_return * np.sqrt(trades_per_year)
        sharpe_ratio = (annualized_mean_return - risk_free_rate) / annualized_std_dev_return

        return sharpe_ratio

    '''
        build_results_dict: utilizes class attributes: 'report_df' and 'portfolio_value' to 
        create a dictionary that summarizes the results of all trades in 'report_df'. The function
        stores this dictionary in the class attributre 'results_dict'.
        ** Function REQUIRES 'build_report_df' to be called before.
    ''' 
    def build_results_dict(self):
        # (1) Divide trades into groups and get information
        w_trades = self.report_df[self.report_df['Net Return'] > 0]
        l_trades = self.report_df[self.report_df['Net Return'] <= 0]
        w_trades_returns = w_trades['Net Return'].sum()
        l_trades_returns = l_trades['Net Return'].sum()
        s1 = self.strategy_obj.Symbol_1
        s2 = self.strategy_obj.Symbol_2
        long_spread_trades  = self.report_df[self.report_df['Long Symbol'] == s1]
        short_spread_trades = self.report_df[self.report_df['Long Symbol'] == s2]
        w_long_spreads  = long_spread_trades[long_spread_trades['Net Return'] > 0]
        w_short_spreads = short_spread_trades[short_spread_trades['Net Return'] > 0]
        date1 = datetime.strptime(self.strategy_obj.start_date, "%Y-%m-%d")
        date2 = datetime.strptime(self.strategy_obj.end_date, "%Y-%m-%d")
        years_difference = abs((date1 - date2).days) / 365.25

        # (2) Compute results for dictionary 
        self.results_dict['Final Portfolio Value']            = self.portfolio_value
        self.results_dict['Net Portfolio Change']             = self.portfolio_value - self.initial_capital
        self.results_dict['Net Returns']                      = ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100
        self.results_dict['Number of Trades']                 = self.report_df.shape[0]
        self.results_dict['Number of Winning Trades']         = w_trades.shape[0]
        self.results_dict['Number of Loosing Trades']         = l_trades.shape[0]
        self.results_dict['Win Rate']                         = (self.results_dict['Number of Winning Trades'] / self.results_dict['Number of Trades']) * 100 if self.results_dict['Number of Trades'] != 0 else None
        self.results_dict['Loss Rate']                        = (self.results_dict['Number of Loosing Trades'] / self.results_dict['Number of Trades']) * 100 if self.results_dict['Number of Trades'] != 0 else None
        self.results_dict['Drawdown']                         = ((self.report_df['Portfolio Value'].max() - self.portfolio_value) / self.report_df['Portfolio Value'].max()) * 100
        self.results_dict['Sharpe Ratio']                     = self.get_sharpe_ratio(years = years_difference)
        self.results_dict['Biggest Winning Trade']            = w_trades['Net Return'].max()
        self.results_dict['Biggest Loosing Trade']            = l_trades['Net Return'].min()
        self.results_dict['Longest Winning Trade']            = w_trades['Duration'].max()
        self.results_dict['Longest Loosing Trade']            = l_trades['Duration'].max()
        self.results_dict['Average Return per Winning Trade'] = w_trades_returns / self.results_dict['Number of Winning Trades'] if self.results_dict['Number of Winning Trades'] != 0 else None 
        self.results_dict['Average Return per Loosing Trade'] = l_trades_returns / self.results_dict['Number of Loosing Trades'] if self.results_dict['Number of Loosing Trades'] != 0 else None
        self.results_dict['Number of Long Spread Trades']     = long_spread_trades.shape[0]
        self.results_dict['Number of Short Spread Trades']    = short_spread_trades.shape[0]
        self.results_dict['Long Spread Win Rate']             = (w_long_spreads.shape[0]  / self.results_dict['Number of Long Spread Trades'])  * 100 if self.results_dict['Number of Long Spread Trades']  != 0 else None
        self.results_dict['Short Spread Win Rate']            = (w_short_spreads.shape[0] / self.results_dict['Number of Short Spread Trades']) * 100 if self.results_dict['Number of Short Spread Trades'] != 0 else None
        self.results_dict['Average Winning Trade Duration']   = w_trades['Duration'].mean()
        self.results_dict['Average Loosing Trade Duration']   = l_trades['Duration'].mean()

        # (3) Compute duration percentiles separately for winning and loosing trades 
        percentiles = [80, 85, 90, 95, 99]
        trade_durations = [sorted(w_trades['Duration']), sorted(l_trades['Duration'])]
        for i in range (0, 2):
            duration = trade_durations[i]
            w_or_l = "Winning" if i == 0 else "Loosing"
            if duration:
                percentile_values = np.percentile(duration, percentiles)
                for p, value in zip(percentiles, percentile_values):
                    self.results_dict[f"Time Taken to Close {p}% of {w_or_l} Trades"] = value
            else:
                for p in percentiles:
                    self.results_dict[f"Time Taken to Close {p}% of {w_or_l} Trades"] = None
        return 
    
    '''
        print_results: prints to terminal all the results of backtest found in results dict 
    '''
    def print_results(self):
        percentage_list = ["Net Returns", "Win Rate", "Loss Rate", "Drawdown", "Biggest Winning Trade", "Biggest Loosing Trade",
                           "Average Return per Winning Trade", "Average Return per Loosing Trade", "Long Spread Win Rate",
                           "Short Spread Win Rate"]
        for key in self.results_dict:
            result  = self.results_dict[key]
            rounded = None if result == None else round(result, 3)
            p = "%" if key in percentage_list else ""
            print(f"{key}: {rounded}{p}")
    
    '''
        backtest: conducts full backtest pipeline, prints results (optional) and 
        returns dictionary with all the results 
    '''
    def backtest(self, show_output = False):
        self.build_report_df()
        self.build_results_dict()
        if show_output == True:
            self.print_results()
        return self.results_dict


################################
# symbol1 = "DHI"
# symbol2 = "LEN"

# z = ZSCORE_STRATEGY(  symbol_1           = symbol1,
#                       symbol_2           = symbol2,

#                       zscore_period      = 20,
#                       zscore_stop_loss   = 2.96,
#                       zscore_take_profit = 0.15,
#                       zscore_entry       = 1.87,  # Z-Score threshold

#                       use_coint_check    = False,
#                       coint_check_length = 0,

#                       use_days_stop      = False,
#                       days_for_stop      = 0,

#                       start_date         = "2018-06-01",
#                       end_date           = "2024-05-01",
#                       path_to_prices_df  = r"C:\Users\seckh\OneDrive\Documents\1.1-RefinitivCleanData.csv"
#                     )

# b = PAIRS_BACKTESTER(strategy_obj                         = z,
#                      initial_capital                      = 10000,
#                      fixed_allocation                     = 0.5,
#                      risk_per_trade                       = 1
#                     )
# b.backtest(show_output = True)
