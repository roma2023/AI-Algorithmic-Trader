from _0_helper_functions import *
from _5_DATA_MANIPULATOR import DATA_MANIPULATOR
'''
                                    << ZSCORE_STRATEGY (module) >>
    => Class Initialization Inputs:
        (1) symbol_1            : string | stock ticker symbol for the first asset
        (2) symbol_2            : string | stock ticker symbol for the second asset
        (3) path_to_prices_df   : string | path to file with OHLCV data of stocks
        (4) start_date          : string | start date in 'YYYY-MM-DD' format
        (5) end_date            : string | end date in 'YYYY-MM-DD' format
        (6) zscore_period       : int    | period used to calculate the Z-score
        (7) zscore_entry        : float  | entry value of Z-score (+/-)
        (8) zscore_stop_loss    : float  | Z-score stop loss (+/-)
        (9) zscore_take_profit  : float  | Z-score take profit (+/-) (usually 0)
        (10) use_coint_check    : bool   | use cointegration check (T/F)
        (11) coint_check_length : int    | number of previous candles for cointegration check
        (12) use_days_stop      : bool   | apply stop loss (T/F)
        (13) days_for_stop      : int    | number of days for stop loss

    ------------ User-facing Methods --------------
        (1) build_backtester_df():
            => Calls methods 'build_general_columns()', 'build_coint_column()' and 'build_signal_column()'
               to build up class attribute 'backtesting_df'.

        (2) change_parameters(symbol_1, symbol_2, zscore_stop_loss, zscore_take_profit, zscore_entry, zscore_period,  
                              use_coint_check, coint_check_length, use_days_stop, days_for_stop):
            => Updates class parameters and rebuilds 'backtesting_df'.
            => Inputs: * same as init of class but without dates        
            => Outputs: None

    ------------ Backend Methods --------------
        (1) build_general_columns():
            => Adds open and close prices for both symbols, spread, and Z-score to class attribute 'backtesting_df'. 

        (2) build_coint_column():
            => Adds a rolling cointegration column to 'backtesting_df' if 'use_coint_check' is True.

        (3) determine_zscore_signal(curr_position, curr_position_entry_day, curr_position_active_days, 
                                    curr_date, prev_zscore, curr_zscore, curr_coint):
            => Determines the current trading signal based on Z-score.
            => Inputs: 
                curr_position             : tuple    | current position in the format (Symbol_1, Symbol_2)
                curr_position_entry_day   : datetime | the date when the current position was entered
                curr_position_active_days : int      | the number of days the current position has been active
                curr_date                 : datetime | the current date
                prev_zscore               : float    | the Z-score from the previous day
                curr_zscore               : float    | the Z-score for the current day
                curr_coint                : int      | the cointegration check result (1 or 0)
            => Outputs: 
                tuple | updated current position, current signal, and current position entry date

        (4) build_signal_column():
            => Adds signal column to 'backtesting_df' indicating trading actions.
'''

class ZSCORE_STRATEGY():
    def __init__(self, 
                 # General Parameters 
                 symbol_1   : str, symbol_2 : str, path_to_prices_df : str, 
                 start_date : str, end_date : str,

                 # Mandatory Parameters 
                 zscore_period    : int  , zscore_entry       : float, 
                 zscore_stop_loss : float, zscore_take_profit : float,
                 
                 # Advanced (Optional) Parameters 
                 use_coint_check : bool, coint_check_length : int, 
                 use_days_stop   : bool, days_for_stop      : int
                 ):
        # (1) Initialize General Parameters 
        self.Symbol_1   = symbol_1
        self.Symbol_2   = symbol_2
        self.start_date = start_date
        self.end_date   = end_date
        self.DATA       = DATA_MANIPULATOR(path_to_prices_df)

        # (2) Initialize Mandatory Parameters
        self.zscore_period      = zscore_period       # Period Used To Calculate the Z-Score
        self.zscore_entry       = zscore_entry        # Entry Value of Z-Score (+/-)
        self.zscore_stop_loss   = zscore_stop_loss    # Z-Score Stop Loss (+/-)
        self.zscore_take_profit = zscore_take_profit  # Z-Score Take Profit (+/-) (usually 0)

        # (3) Initialize Advanced (optional) Parameters 
        self.use_coint_check    = use_coint_check     # Use Cointegration Check (T/F)
        self.coint_check_length = coint_check_length  # Number of Previous Candles for Cointegration Check

        self.use_days_stop      = use_days_stop       # Apply Stop Loss (T/F)
        self.days_for_stop      = days_for_stop + 1   # Number of Days for Stop Loss

        # (4) Initialze Output Variable
        self.init = False
        self.backtesting_df = pd.DataFrame()

    '''
        build_general_columns: builds columns displyed below and stores results in 
        class attribute 'backtesting_df'.
        
        | Open_Symbol_1 | Close_Symbol_1 | Open_Symbol_2 | Close_Symbol_2 | Spread | Z-Score | 
    '''
    def build_general_columns(self):
        # (1) Initalize Data From Backtest Obj
        if self.DATA.is_initialized() == False:
            self.DATA.initialize_data()

        # (2) Get data from specified period for both symbols using object from 'DATA_MANIPULATOR' module
        self.Symbol_1_prices = self.DATA.get_data_between_dates(symbol = self.Symbol_1, start_date = self.start_date, end_date = self.end_date)
        self.Symbol_2_prices = self.DATA.get_data_between_dates(symbol = self.Symbol_2, start_date = self.start_date, end_date = self.end_date)

        # (3) Build opens & closes for symbols 
        self.backtesting_df[f"Open_{self.Symbol_1}"]  = self.Symbol_1_prices["Open"]    # OPEN_1
        self.backtesting_df[f"Close_{self.Symbol_1}"] = self.Symbol_1_prices["Close"]   # CLOSE_1
        self.backtesting_df[f"Open_{self.Symbol_2}"]  = self.Symbol_2_prices["Open"]    # OPEN_2
        self.backtesting_df[f"Close_{self.Symbol_2}"] = self.Symbol_2_prices["Close"]   # CLOSE_2

        # (4) Build spread and Z-score
        hedge_ratio = self.DATA.calculate_hedge_ratio(self.Symbol_1_prices["Close"], self.Symbol_2_prices["Close"])
        spread = self.DATA.calculate_spread(self.Symbol_1_prices["Close"], self.Symbol_2_prices["Close"], hedge_ratio)
        zscore = self.DATA.calculate_zcore(spread, self.zscore_period)
        self.backtesting_df["Spread"]      = spread.tolist()       # SPREAD
        self.backtesting_df["Z-Score"]     = zscore                # ZSCORE
        
        return 

    '''
        build_coint_column: adds rolling cointegration column to class attribute 'backtesting_df'
        if the 'use_coint_check' class parameter was set to true.
    '''
    def build_coint_column(self):
        if self.use_coint_check:
            rolling_cointegration = self.DATA.calculate_rolling_cointegration(close_prices_1 = self.Symbol_1_prices["Close"],
                                                                              close_prices_2 = self.Symbol_2_prices["Close"], 
                                                                              window         =self.coint_check_length)
            self.backtesting_df["Coint_Check"] = rolling_cointegration 
    
    ''' 
        determine_zscore_signal: returns new current position, current signal 
        and current position (entry) date. 
    '''
    def determine_zscore_signal(self, curr_position, curr_position_entry_day, curr_position_active_days, 
                                curr_date, prev_zscore, curr_zscore, curr_coint):
            # Case 1: Active Position
            if (curr_position != None): 
                tp = self.zscore_take_profit if curr_position == ("BUY", "SELL") else -self.zscore_take_profit
                
                # (i)  Ensure no exit signal has been triggered
                if (
                    (crossover(prev_zscore, curr_zscore, tp) == True)                                  or   # (1) ZSCORE CROSSES 0 (or tp)
                    (curr_position_active_days >= self.days_for_stop and self.use_days_stop == True)   or   # (2) POSITION HAS EXCEDED THE USER'S STOP DAYS
                    (curr_position == ("BUY", "SELL") and (curr_zscore < -self.zscore_stop_loss))      or   # (3) LONG ON SPREAD  & ZSCORE BECOMES MORE NEGATIVE THAN SL THRESHOLD
                    (curr_position == ("SELL", "BUY") and (curr_zscore > self.zscore_stop_loss))            # (4) SHORT ON SPREAD & ZSCORE BECOMES MORE POSITIVE THAN SL THRESHOLD
                   ):         

                    curr_position      = None
                    signal             = ("CLOSE", "CLOSE")
                    curr_position_date = None

                 # (ii) If no exit signal has been triggered, hold position
                else:
                    curr_position = curr_position
                    signal = ("HOLD", "HOLD")
                    curr_position_date = curr_position_entry_day
            
            # Case 2: No Active Positons, look for entry signals 
            # (i)  Check for Long Spread: (Buy Symbol 1, Sell Symbol 2)
            elif (
                  (curr_zscore < -self.zscore_entry)      and
                  (curr_zscore > -self.zscore_stop_loss)  and
                  (curr_position == None)                 and
                  (curr_coint    == 1)
                 ):
                curr_position      = ("BUY", "SELL")
                signal             = ("BUY", "SELL")
                curr_position_date = curr_date
            
            # (ii) Check for Short Spread: (Sell Symbol 1, Buy Symbol 2)
            elif (
                  (curr_zscore > self.zscore_entry)      and
                  (curr_zscore < self.zscore_stop_loss)  and
                  (curr_position == None)                and
                  (curr_coint    == 1)
                 ):
                curr_position      = ("SELL", "BUY")
                signal             = ("SELL", "BUY")
                curr_position_date = curr_date
            
            # (3) No Signal.
            else:
                curr_position      = None
                signal             = ("NONE", "NONE")
                curr_position_date = None
            
            return curr_position, signal, curr_position_date
    
    '''
        change_paramters: takes in all the init parameters and reasign them to the class building 
        a new backtesting_df 
    '''
    def change_parameters(self, symbol_1, symbol_2, 
                          zscore_stop_loss, zscore_take_profit, zscore_entry, zscore_period,  
                          use_coint_check, coint_check_length, 
                          use_days_stop, days_for_stop):
        # (i) Update symbols & cointegration information and recalculate columns if they changed
        prev_symbol_1        = self.Symbol_1
        prev_symbol_2        = self.Symbol_2
        prev_coint_check     = self.use_coint_check
        prev_coint_check_len = self.coint_check_length
        prev_zscore_period   = self.zscore_period 

        self.Symbol_1           = symbol_1
        self.Symbol_2           = symbol_2
        self.use_coint_check    = use_coint_check
        self.coint_check_length = coint_check_length
        self.zscore_period      = zscore_period

        if (prev_symbol_1 != self.Symbol_1) or (prev_symbol_2 != self.Symbol_2) or (prev_zscore_period != self.zscore_period):
            self.build_general_columns()
            self.build_coint_column()
        else:
            if (prev_coint_check != self.use_coint_check) or (prev_coint_check_len != self.coint_check_length):
                self.build_coint_column()

        # (ii) Update ZScore Parameters & Rebuild Signal Column
        self.zscore_period      = zscore_period
        self.zscore_entry       = zscore_entry
        self.zscore_stop_loss   = zscore_stop_loss
        self.zscore_take_profit = zscore_take_profit
        self.use_days_stop      = use_days_stop
        self.days_for_stop      = days_for_stop
        self.build_signal_column()
        
        return 
        
    '''
        build_signal_column: adds signal column to backtesting_df where each entry in the column is 
            => (BUY, SELL)    - Long on Spread 
            => (SELL, BUY)    - Short on Spread 
            => (HOLD, HOLD)   - Hold the current position we are in 
            => (CLOSE, CLOSE) - Close the current position we are in 
            => (NONE, NONE)   - Do nothing 
        
        | Open_Symbol_1 | Close_Symbol_1 | Open_Symbol_2 | Close_Symbol_2 | Spread | Z-Score | Coint (Optional) |Signal |

    '''
    def build_signal_column(self):
        # (1) Output Variable
        signal_column             = []

        # (2) Tracker Variables
        curr_position             = None  # Tupple: ("BUY"/"SELL" >> Symbol 1,  "BUY"/"SELL" >> Symbol 2)
        curr_position_date        = None
        curr_position_active_days = 0
        prev_zscore               = 0

        for index, row in self.backtesting_df.iterrows():
            # (1) Get current zscore & date
            curr_zscore = row["Z-Score"]
            curr_date   = index

            # (2) Get current cointegration value (if requested)
            curr_coint = 1 
            if self.use_coint_check == True:
                curr_coint = row["Coint_Check"]
                try:
                    curr_coint = int(curr_coint)
                except:
                    curr_coint = 0

            # (3) Get current time of active position (if requested by user and position is open)
            if (curr_position != None) and (self.use_days_stop == True):
                delta = curr_date - curr_position_date
                curr_position_active_days = delta.days

            # (4) Determine signal and update tracker variables
            curr_position, signal, curr_position_date = self.determine_zscore_signal( curr_position             = curr_position, 
                                                                                      curr_position_active_days = curr_position_active_days, 
                                                                                      curr_position_entry_day   = curr_position_date,
                                                                                      curr_date                 = curr_date, 
                                                                                      prev_zscore               = prev_zscore, 
                                                                                      curr_zscore               = curr_zscore, 
                                                                                      curr_coint                = curr_coint)
            # (5) Add signal to 'signal_column' array 
            signal_column.append(signal)

            # (6) Make previous Z-score equal current Z-score for next iteration
            prev_zscore = curr_zscore
        
        # (7) Assign signal column to 'backtesting_df' class attribute 
        self.backtesting_df["Signal"] = signal_column
        return  
    
    '''
        build_backtester_df: calls class methods to create 'backtesting_df'
    '''
    def build_backtester_df(self):
        if self.init == False:
            self.init = True
            self.build_general_columns()
            self.build_coint_column()
            self.build_signal_column()

        return self.backtesting_df
