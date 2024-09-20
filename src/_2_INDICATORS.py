from _0_helper_functions import *
import ta 
from ta.volume import EaseOfMovementIndicator
from ta.volume import ChaikinMoneyFlowIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import StochRSIIndicator
from ta.momentum import ROCIndicator
from ta.momentum import WilliamsRIndicator
from ta.momentum import UltimateOscillator
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.trend import ADXIndicator
from ta.trend import CCIIndicator
from ta.others import DailyLogReturnIndicator
from ta.others import DailyReturnIndicator
"""
                                     << INDICATORS (module) >>
    => Class Initialization Inputs: *only needs one of the two parameters*
        (1) path_to_clean_data_csv : str        | path to the CSV file containing clean data
        (2) clean_data_df          : data frame | CSV containing clean data 

    ------------ User-facing Methods --------------
    *** all indicator methods add columns to the class attribute 'main_df' *** 

        (1) initializeMainDataFrame():
            => Initializes the main DataFrame with data from the CSV file and extracts global tickers.
            => Outputs:
                None

        (2) addRSI(period):
            => Adds the RSI indicator to the DataFrame.
            => Inputs:
                period : int | period for the RSI calculation
            => Outputs:
                list | titles of the added RSI columns

        (3) addStochastic(k_length, k_smoothing, d_smoothing):
            => Adds the Stochastic Oscillator indicator to the DataFrame.
            => Inputs:
                k_length   : int | period for %K calculation
                k_smoothing: int | smoothing period for %K
                d_smoothing: int | smoothing period for %D
            => Outputs:
                tuple | titles of the added Stochastic Oscillator columns

        (4) addStochRSI(period, ma_period, ma_k):
            => Adds the StochRSI indicator to the DataFrame.
            => Inputs:
                period  : int | period for RSI calculation
                ma_period: int | moving average period for %K
                ma_k    : int | moving average period for %D
            => Outputs:
                tuple | titles of the added StochRSI columns

        (5) addROC(period):
            => Adds the ROC indicator to the DataFrame.
            => Inputs:
                period : int | period for ROC calculation
            => Outputs:
                list | titles of the added ROC columns

        (6) addWilliamsR(period):
            => Adds the Williams %R indicator to the DataFrame.
            => Inputs:
                period : int | period for Williams %R calculation
            => Outputs:
                list | titles of the added Williams %R columns

        (7) addUO(short_period, mid_period, long_period, w1=4.0, w2=2.0, w3=1.0):
            => Adds the Ultimate Oscillator indicator to the DataFrame.
            => Inputs:
                short_period: int  | short period for UO calculation
                mid_period  : int  | mid period for UO calculation
                long_period : int  | long period for UO calculation
                w1          : float| weight for short period (default 4.0)
                w2          : float| weight for mid period (default 2.0)
                w3          : float| weight for long period (default 1.0)
            => Outputs:
                str | title of the added UO column

        (8) addVWAP(period):
            => Adds the VWAP indicator to the DataFrame.
            => Inputs:
                period : int | period for VWAP calculation
            => Outputs:
                list | titles of the added VWAP columns

        (9) addCMF(period):
            => Adds the Chaikin Money Flow indicator to the DataFrame.
            => Inputs:
                period : int | period for CMF calculation
            => Outputs:
                list | titles of the added CMF columns

        (10) addEMV(period):
            => Adds the Ease of Movement indicator to the DataFrame.
            => Inputs:
                period : int | period for EMV calculation
            => Outputs:
                list | titles of the added EMV columns

        (11) addBB(period, std=2):
            => Adds the Bollinger Bands indicator to the DataFrame.
            => Inputs:
                period : int   | period for BB calculation
                std    : float | standard deviation (default 2)
            => Outputs:
                list | titles of the added Bollinger Bands columns

        (12) addATR(period):
            => Adds the Average True Range indicator to the DataFrame.
            => Inputs:
                period : int | period for ATR calculation
            => Outputs:
                list | titles of the added ATR columns

        (13) addMA(ema_or_sma, period):
            => Adds the Moving Average (EMA or SMA) indicator to the DataFrame.
            => Inputs:
                ema_or_sma: str | type of moving average ("EMA" or "SMA")
                period    : int | period for MA calculation
            => Outputs:
                list | titles of the added MA columns

        (14) addMACD(fast, slow, sign):
            => Adds the MACD indicator to the DataFrame.
            => Inputs:
                fast: int | fast period for MACD
                slow: int | slow period for MACD
                sign: int | signal period for MACD
            => Outputs:
                tuple | titles of the added MACD columns

        (15) addADX(period):
            => Adds the Average Directional Movement Index indicator to the DataFrame.
            => Inputs:
                period : int | period for ADX calculation
            => Outputs:
                list | titles of the added ADX columns

        (16) addCCI(period, c=0.015):
            => Adds the Commodity Channel Index indicator to the DataFrame.
            => Inputs:
                period : int   | period for CCI calculation
                c      : float | constant (default 0.015)
            => Outputs:
                list | titles of the added CCI columns

        (17) addFibLvls(period):
            => Adds Fibonacci Retracement levels to the DataFrame.
            => Inputs:
                period : int | period for Fibonacci levels calculation
            => Outputs:
                list | titles of the added Fibonacci levels columns

        (18) addHighsAndLows(period):
            => Adds high and low prices to the DataFrame.
            => Inputs:
                period : int | period for high and low prices calculation
            => Outputs:
                list | titles of the added high and low columns

        (19) addBullBear(ema_period):
            => Adds Bull and Bear Power indicators to the DataFrame.
            => Inputs:
                ema_period : int | period for EMA calculation used in Bull/Bear Power
            => Outputs:
                list | titles of the added Bull/Bear Power columns

        (20) addDailyReturns():
            => Adds Daily Log Return and Daily Return indicators to the DataFrame.
            => Outputs:
                list | titles of the added Daily Returns columns

        (21) addCumulativeReturns():
            => Adds Cumulative Returns to the DataFrame.
            => Outputs:
                list | titles of the added Cumulative Returns columns
"""
class INDICATORS():
    def __init__(self, path_to_clean_data_csv : str = None, clean_data_csv = None):
        self.path_to_csv   = path_to_clean_data_csv
        self.main_df       = clean_data_csv
        self.globalTickers = []    # All Tickers of Data Frame

        self.momentum_and_trend    = []
        self.volume_and_volatility = []
        self.other_indicators      = []
    
    def return_df_with_indicators(self):
        return self.main_df
    
    def initializeMainDataFrame(self):
        if self.path_to_csv != None:
            self.main_df       = pd.read_csv(self.path_to_csv)
        
        if 'Unnamed: 0' in self.main_df:
            self.main_df.drop(columns = 'Unnamed: 0', inplace = True)

        self.globalTickers = getTickers(self.main_df)
    # ============================================= MOMENTUM INDICATORS ============================================= #
    # FUNCTION TO ADD RSIs
    def addRSI(self, period):
        titleOfRSI = 'RSI_'+ str(period)
        self.main_df [titleOfRSI] = ta.momentum.RSIIndicator(close = self.main_df ['close'], window = period).rsi()
        self.main_df = periodCorrector(self.main_df, period, titleOfRSI, self.globalTickers)
        return [titleOfRSI]    
    
    # FUNCTION TO ADD STOCHASTIC OSCILATOR INDICATOR
    def addStochastic(self, k_length, k_smoothing, d_smoothing):
        # Calculate rolling lowest low and highest high
        lowest_low = 'Lowest_Low_' + str(k_length)
        highest_high = 'Highest_High_' + str(k_length)
        K = f"%K_({k_length},{k_smoothing},{d_smoothing})"
        D = f"%D_({k_length},{k_smoothing},{d_smoothing})"


        self.main_df[lowest_low]   = self.main_df['low'] .rolling(window = k_length).min()
        self.main_df[highest_high] = self.main_df['high'].rolling(window = k_length).max()

        # Calculate %K
        self.main_df[K] = 100 * ((self.main_df['close'] - self.main_df[lowest_low]) / (self.main_df[highest_high] - self.main_df[lowest_low]))

        # Smooth %K
        self.main_df[K] = self.main_df[K].rolling(window = k_smoothing).mean()

        # Calculate %D
        self.main_df[D] = self.main_df[K].rolling(window = d_smoothing).mean()

        # Drop Lowest Low and Highest High
        self.main_df.drop([lowest_low, highest_high], axis = 1, inplace = True)

        '''
        Correct periords, for %K we need k_length + k_smoothing, for
        %D we need k_length + k_smoothing + d_smoothing days and for lowest low
        and highest high we just need 20 k_lenght days
        '''
        self.main_df = periodCorrector(self.main_df, k_length - 1 + k_smoothing - 1, K, self.globalTickers)
        self.main_df = periodCorrector(self.main_df, k_length - 1 + k_smoothing - 1 + d_smoothing - 1, D, self.globalTickers)

        return K, D

    # FUNCTION TO ADD STOCHASTIC_RSI INDICATOR
    def addStochRSI(self, period, ma_period, ma_k):
        rsi = f'StochRSI_({period}, {ma_period}, {ma_k})_rsi'
        d = f'StochRSI_({period}, {ma_period}, {ma_k})_%D'
        k = f'StochRSI_({period}, {ma_period}, {ma_k})_%K'


        stochRSI = StochRSIIndicator( close  =  self.main_df ['close'],
                                      window = period,
                                      smooth1= ma_period,
                                      smooth2= ma_k
                                    )
        self.main_df [rsi] = stochRSI.stochrsi()
        self.main_df [d]   = stochRSI.stochrsi_d()
        self.main_df [k]   = stochRSI.stochrsi_k()

        '''
            Correct the period. The stoch_rsi needs period x 2 days, the k line needs ]
            period x 2 + ma_period and then the d line needs period x 2 + ma_period + ma_k
        '''
        self.main_df  = periodCorrector(self.main_df , period * 2 - 1, rsi, self.globalTickers)
        self.main_df  = periodCorrector(self.main_df , period * 2 + ma_period - 2, k, self.globalTickers)
        self.main_df  = periodCorrector(self.main_df , period * 2 + ma_period + ma_k - 3, d, self.globalTickers)

        return rsi, d, k

    # FUNCTION TO ADD RATE OF CHANGE (ROC) INDICATOR
    def addROC(self, period):
        title = f'ROC_{period}'
        roc = ROCIndicator(close = self.main_df ['close'], window = period, fillna = True)
        self.main_df [title] = roc.roc()
        self.main_df  = periodCorrector(self.main_df , period + 1, title, self.globalTickers)
        return [title] 

    # FUNCTION TO ADD WILLIAMS %R INDICATOR
    def addWilliamsR(self, period):
        title = f'Williams %R_{period}'
        wr = WilliamsRIndicator(high  = self.main_df ['high'],
                                low   = self.main_df ['low'],
                                close = self.main_df ['close'],
                                lbp   = period,
                                fillna= False)
        self.main_df [title] = wr.williams_r()
        self.main_df  = periodCorrector(self.main_df , period, title, self.globalTickers)
        return [title] 
    
    # FUNCTION TO ADD ULTIMATE OSCILLATOR (UO) INDICATOR
    def addUO(self, short_period, mid_period, long_period, w1 = 4.0, w2 = 2.0, w3 = 1.0):
        title = f'UO_({short_period},{mid_period},{long_period})'
        uo = UltimateOscillator(high   = self.main_df ['high'],
                                low    = self.main_df ['low'],
                                close  = self.main_df ['close'],
                                window1= short_period,
                                window2= mid_period,
                                window3= long_period,
                                weight1= w1,
                                weight2= w2,
                                weight3= w3
                                )
        self.main_df [title] = uo.ultimate_oscillator()
        periodToCorrect = max([short_period, mid_period, long_period]) + 1
        self.main_df = periodCorrector(self.main_df , periodToCorrect, title, self.globalTickers)
        return title
    
    # ============================================= VOLUME INDICATORS ============================================= #
    # FUNCTION TO ADD VOLUME WEIGHTED AVERAGE PRICE
    def addVWAP(self, period):
        titleVWAP = 'VWAP_' + str(period)
        vwap = VolumeWeightedAveragePrice(high    = self.main_df['high'],
                                          low    = self.main_df['low'],
                                          close  = self.main_df['close'],
                                          volume = self.main_df['volume'],
                                          window = period
                                        )
        self.main_df[titleVWAP] = vwap.volume_weighted_average_price()
        self.main_df = periodCorrector(self.main_df, 14, titleVWAP, self.globalTickers)
        return [titleVWAP]   

    # ADD CHAIKIN MONEY FLOW (CMF) INDICATOR
    def addCMF(self, period):
        title = f'CNF_{period}'
        cmf = ChaikinMoneyFlowIndicator(high   = self.main_df['high'],
                                        low    = self.main_df['low'],
                                        close  = self.main_df['close'],
                                        volume = self.main_df['volume'],
                                        window = period)
        self.main_df[title] = cmf.chaikin_money_flow()
        self.main_df = periodCorrector(self.main_df, period, title, self.globalTickers)
        return [title]

    # ADD EASE OF MOVEMENT (EMV) INDICATOR
    def addEMV(self, period):
        title = f'EMV_{period}'
        emv = EaseOfMovementIndicator(high   = self.main_df['high'],
                                      low    = self.main_df['low'],
                                      volume = self.main_df['volume'],
                                      window = period
                                     )
        self.main_df[title] = emv.ease_of_movement()
        self.main_df = periodCorrector(self.main_df, period, title, self.globalTickers)
        return [title]
    # ============================================= VOLATILITY INDICATORS ============================================= #
    # FUNCTION TO ADD BOLLINGER BANDS
    def addBB(self, period, std = 2):
        # Name for the columns
        average_line_title = 'BB_avrg_'   + str(period) + '_' + str(std)
        percent_line_title = 'BB_%_band_' + str(period) + '_' + str(std)
        width_line_title   = 'BB_w_band_' + str(period) + '_' + str(std)

        high_line_title = 'BB_high_' + str(period) + '_' + str(std)
        high_line_indic = 'BB_high_i_' + str(period) + '_' + str(std)

        low_line_title = 'BB_low_' + str(period) + '_' + str(std)
        low_line_indic = 'BB_low_i' + str(period) + '_' + str(std)

        # Initialize Indicator
        bb = BollingerBands(close = self.main_df["close"], window = period, window_dev = std)

        # Insert New Columns
        self.main_df[average_line_title] = bb.bollinger_mavg()
        self.main_df[percent_line_title] = bb.bollinger_pband()
        self.main_df[width_line_title]   = bb.bollinger_wband()

        self.main_df[low_line_title] = bb.bollinger_lband()
        self.main_df[low_line_indic] = bb.bollinger_lband_indicator()

        self.main_df[high_line_title] = bb.bollinger_hband()
        self.main_df[high_line_indic] = bb.bollinger_hband_indicator()

        # Correct Columns like other indicators
        self.main_df = periodCorrector(self.main_df, period, average_line_title, self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period, percent_line_title, self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period, width_line_title,   self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period, low_line_title,     self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period, low_line_indic,     self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period, high_line_title,    self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period, high_line_indic,    self.globalTickers)

        return [average_line_title, percent_line_title, width_line_title, high_line_title, 
                high_line_indic, low_line_title, low_line_indic]
    
    # FUNCTION TO ADD ATR (AVERAGE TRUE RANGE)
    def addATR(self, period):
        atr_title = 'ATR_'+ str(period)
        self.main_df[atr_title] = ta.volatility.AverageTrueRange( high  = self.main_df['high'],
                                                                  low   = self.main_df['low'],
                                                                  close = self.main_df['close'],
                                                                  window = period).average_true_range()
        self.main_df = periodCorrector(self.main_df, period, atr_title, self.globalTickers)
        return [atr_title]
    
    # ============================================= TREND INDICATORS ============================================= #
    # FUNCTION TO ADD MOVING AVERAGES
    def addMA(self, ema_or_sma, period):
        titleOfMA = ema_or_sma + "_" + str(period)

        if ema_or_sma == 'EMA':
            self.main_df[titleOfMA] = ta.trend.ema_indicator(self.main_df['close'], window = period)

        elif ema_or_sma == 'SMA':
            self.main_df[titleOfMA] = ta.trend.sma_indicator(self.main_df['close'], window = period)

        else:
            print("ERROR: input EMA or SMA")

        self.main_df = periodCorrector(self.main_df, period, titleOfMA, self.globalTickers)
        return [titleOfMA]
    
    # FUNCTION TO ADD MACD
    def addMACD(self, fast, slow, sign):
        mainStr = 'MACD_(' + str(fast) + ',' + str(slow) + ','+ str(sign) + ')'
        macd = MACD(close = self.main_df['close'], window_slow = slow, window_fast = fast, window_sign = sign)
        macd_line = mainStr + '_macdLine'
        macd_histogram = mainStr + '_hist'
        macd_signal = mainStr + '_signalLine'

        self.main_df[macd_line] = macd.macd()
        self.main_df[macd_histogram] = macd.macd_diff()
        self.main_df[macd_signal] = macd.macd_signal()

        '''Correct the Periods. The MACD line requires 'slow' days of data to start calculating
            while the signal line & histogram requries 'slow' + 'sign' days of data to start
            calculating.
        '''
        self.main_df = periodCorrector(self.main_df, slow       , macd_line,      self.globalTickers)
        self.main_df = periodCorrector(self.main_df, slow + sign, macd_histogram, self.globalTickers)
        self.main_df = periodCorrector(self.main_df, slow + sign, macd_signal,    self.globalTickers)

        return macd_line, macd_histogram, macd_signal
    
    # FUNCTION TO ADD AVERAGE DIRECTIONAL MOVEMENT INDEX (ADX) INDICATOR
    def addADX(self, period):
        # Titles
        adx_avrg  = f'ADX_avrg_{period}'
        adx_minus = f'ADX_minus_{period}'
        adx_pos   = f'ADX_pos_{period}'

        # Create Indicator
        adx = ADXIndicator( high   = self.main_df['high'],
                            low    = self.main_df['low'],
                            close  = self.main_df['close'],
                            window = period,
                            fillna = False)
        self.main_df[adx_avrg]  = adx.adx()
        self.main_df[adx_minus] = adx.adx_neg()
        self.main_df[adx_pos]   = adx.adx_pos()

        self.main_df = periodCorrector(self.main_df, period + 2, adx_minus, self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period + 2, adx_pos,   self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period * 2, adx_avrg,  self.globalTickers)

        # Create Title 
        return [adx_avrg, adx_minus, adx_pos]
    
    # FUNCTION TO ADD COMMODITY CHANNEL INDEX (CCI) INDICATOR
    def addCCI(self, period, c = 0.015):
        title = f'CCI_{period}_c= {c}'
        cci = CCIIndicator( high     = self.main_df['high'],
                            low      = self.main_df['low'],
                            close    = self.main_df['close'],
                            window   = period,
                            constant = c)
        self.main_df[title] = cci.cci()
        self.main_df = periodCorrector(self.main_df, period, title, self.globalTickers)
        return [title]
    
    # ============================================= OTHER INDICATORS ============================================= #
    # FUNCTION TO ADD FIBONNACI RETRACEMENT LEVELS
    def addFibLvls(self, period):
        # Calculate rolling lowest low, highest high & range
        lowest_low   = 'Lowest_Low_' + str(period)
        highest_high = 'Highest_High_' + str(period)
        range        = 'Range_' + str(period)

        self.main_df[lowest_low]   = self.main_df['low' ].rolling(window = period).min()
        self.main_df[highest_high] = self.main_df['high'].rolling(window = period).max()
        self.main_df[range]        = self.main_df[highest_high] - self.main_df[lowest_low]

        # Calculate lvls: retracement level = high - (range x fib ratio)
        fib_ratios = [(0.236, "23.6%"), (0.382, "38.2%"), (0.5, "50%"),
                      (0.618, "61.8%"), (0.786, "78.6%"), (1.0, "100%")]
        
        titles = []
        for (ratio, percent) in fib_ratios:
            title = percent + f'_({period})'
            titles.append(title)
            self.main_df[title] = self.main_df[highest_high] - (self.main_df[range] * ratio)
            self.main_df = periodCorrector(self.main_df, period, title, self.globalTickers)

        # Drop Range, Highest High and Lowest Low columns
        self.main_df.drop([lowest_low, highest_high, range], axis = 1, inplace = True)

        # Return List of Titles 
        return titles
    
    # FUNCTION TO ADD HIGHS AND LOWS
    def addHighsAndLows(self, period):
        high = f'High_{period}'
        low  = f'Low_{period}'

        self.main_df[high] = self.main_df['high'].rolling(window = period).max()
        self.main_df[low]  = self.main_df['low' ].rolling(window = period).min()

        self.main_df = periodCorrector(self.main_df, period, high, self.globalTickers)
        self.main_df = periodCorrector(self.main_df, period, low,  self.globalTickers)
        
        return [high, low]

    # FUNCTION TO ADD BULL/BEAR POWER INDICATOR
    def addBullBear(self, ema_period):
        bull = f'BullPower_({ema_period})' # Daily High - currEMA
        bear = f'BearPower_({ema_period})' # Daily Low  - currEMA

        # Add a temporary ema to our chart
        title = self.addMA("EMA", ema_period)[0]

        # Do the calculations
        self.main_df[bull] = self.main_df['high'] - self.main_df[title]
        self.main_df[bear] = self.main_df['low' ] - self.main_df[title]

        # Drop temporary ema
        self.main_df.drop(title, axis = 1)

        return [bull, bear] 

    def addDailyReturns(self):
        # Add Daily Log Return Indicator
        DLR = DailyLogReturnIndicator(close = self.main_df['close'])
        self.main_df['Log Return'] = DLR.daily_log_return()

        # Add Daily Return Indicator
        DR = DailyReturnIndicator(close = self.main_df['close'])
        self.main_df['Return'] = DR.daily_return()

        # Correct Period because we need the previous day for calculation
        self.main_df = periodCorrector(self.main_df, 2,'Log Return', self.globalTickers)
        self.main_df = periodCorrector(self.main_df, 2,'Return',     self.globalTickers)

        # Create Title 
        return ['Log Return', 'Return'] 
    
    # FUNCTION TO ADD CUMULATIVE RETURNS TO ALL SYMBOLS
    def addCumulativeReturns(self):
        # Create a new column for cumulative returns initialized to NaN
        self.main_df['Cumulative Return'] = float('nan')

        # Group the DataFrame by ticker
        grouped = self.main_df.groupby('ticker')

        for ticker in self.globalTickers:
            # Get all the data for that ticker
            if ticker in grouped.groups:
                ticker_rows = grouped.get_group(ticker)

                # Get the initial price of that ticker
                P_initial = ticker_rows['close'].iloc[0]

                # Calculate the cumulative return and assign it to the new column
                cumulative_return = (ticker_rows['close'] / P_initial - 1) * 100
                self.main_df.loc[self.main_df ['ticker'] == ticker, 'Cumulative Return'] = cumulative_return.values

        # Correct Period
        self.main_df = periodCorrector(self.main_df , 2, 'Cumulative Return', self.globalTickers)

        return ["Cumulative Return"]