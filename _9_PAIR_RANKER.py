from _0_helper_functions import *
from _5_DATA_MANIPULATOR import DATA_MANIPULATOR
'''
                                            << PAIR_RANKER (module) >>
    => Class Initialization Inputs:
        (1) path_to_prices_df : str | path to file with OHLCV data of stocks
    
    ------------ User-facing Methods --------------
        (1) rank_pairs(path_to_pairs_csv : str, start_date : str, end_date : str, weights_per_col : dict):
            => Ranks pairs from best to worst using specified weights for each statistical metric.
            => Inputs:
                - path_to_pairs_csv: str | path to csv file with pairs in intersection, union and differences (exclusive 1 and 2)
                - start_date: str        | start date for data
                - end_date: str          | end date for data
                - weights_per_col: dict  | dictionary of weights for each statistical metric
            => Output:
                - dataframe | best pair at the top.

    ------------ Backend Methods --------------
        (1) get_hypothesis_level(p_value : float, adf_value : float):
            => Returns the significance level of mean reversion hypothesis based on p-value and adf-value.
            => Null Hypothesis: The spread is non-stationary (does not mean revert)
            => Alternative Hypothesis: The spread is stationary (mean revert)
            =>  Significance Level: The probability of rejecting the null hypothesis when it is true
                - i.e the probability the spread is non-mean reverting when we said it was
                - Level 3 (Safest)    : p value < 0.01 and ADF Value < -3.43
                - Level 2 (Safe)      : p value < 0.05 and ADF Value < -2.86
                - Level 1 (Risky)     : p value < 0.10 and ADF Value < -2.57
                - Level 0 (Dangerous) : p value > 0.10 and ADF Value > -2.57

        (2) get_pair_half_life(spread):
            => Calculates the half-life of mean reversion for a pair's spread.
            => Inputs:
                - spread: array-like | the spread of the pair
            => Outputs:
                - float | half life duration if mean reversion exists, None otherwise.

        (3) get_pair_stats(symbol_1 : str, symbol_2 : str, start_date : str, end_date : str):
            => Returns a dictionary for the pair created by the two symbols passed 
               that contains the pair's ADF value, p-value, significance level, half-life duration and 
               number of times spread has crossed the 0 line. 
            => Inputs:
                - symbol_1: str   | ticker symbol for the first asset
                - symbol_2: str   | ticker symbol for the second asset
                - start_date: str | start date for data
                - end_date: str   | end date for data

        (4) get_pair_and_strength_from_str(pair_entry : str):
            => Helper function to parse string entry of union-intersection-difference df
            => Outputs:
                - pair: str       | pair identifier
                - strength: float | correlation strength
                - symbols: list   | list of two symbols
            => Example:
                "('DHI-LEN', 54.0)" => 'DHI-LEN', 54.0, ['DHI', 'LEN']

        (5) get_all_pairs_stats(path_to_pairs_csv : str, start_date : str, end_date : str):
            => Returns dataframe with pairs & their statistical arbitrage values.
            => Inputs:
                - path_to_pairs_csv: str | path to csv file with pairs
                - start_date: str        | start date for data
                - end_date: str          | end date for data
            => Output:
                - dataframe with columns: Pair, Correlational Strength, Half Life, ADF Value, P-Value, Mean Reversion Significance Level, Group.
'''

class PAIR_RANKER():
    def __init__(self, path_to_prices_df : str):
        # (1) Initialize parameters:
        self.path_to_prices_df = path_to_prices_df

        # (2) Initialize intermediaries variables:
        self.DATA = DATA_MANIPULATOR(self.path_to_prices_df)
    
    '''
        get_hypothesis_level: returns the significance level of mean reversion hypothesis 
        based on p-value and adf-value.
            => Null Hypothesis: The spread is non-stationary (does not mean revert)
            => Alternative Hypothesis: The spread is stationary (mean revert)
            =>  Significance Level: The probability of rejecting the null hypothesis when it is true
                - i.e the probability the spread is non-mean reverting when we said it was
                - Level 3 (Safest)    : p value < 0.01 and ADF Value < -3.43
                - Level 2 (Safe)      : p value < 0.05 and ADF Value < -2.86
                - Level 1 (Risky)     : p value < 0.10 and ADF Value < -2.57
                - Level 0 (Dangerous) : p value > 0.10 and ADF Value > -2.57
    '''
    def get_hypothesis_level(self, p_value : float , adf_value : float):
        level = 0
        if p_value < 0.01 and adf_value < -3.43:
            level = 3
        elif p_value < 0.05 and adf_value < -2.86:
            level = 2
        elif p_value < 0.10 and adf_value < -2.57:
            level = 1
        return level
    
    '''
        get_pair_half_life: calculates the half life of mean reversion for a pair's spread
    '''
    def get_half_life(self, spread):
        lagged_spread = np.roll(spread, shift = 1)         # (i)    Lag the spread by one day
        lagged_spread[0] = 0                               # (ii)   Remove 1st element of lagged
        spread_diff = spread - lagged_spread               # (iii)  Get list of daily spread differences (today's spread - yesterday's spread)
        spread_diff = spread_diff[1:]                      # (iv)   Remove 1st element (NaN) from daily differences
        lagged_spread = lagged_spread[1:]                  # (v)    Remove 1st element (that 0) from lagged spread
        lagged_spread = lagged_spread.reshape(-1, 1)       # (vi)   Reshape lagged spread for OLS
        regression = OLS(spread_diff, lagged_spread).fit() # (vii)  Fit regression model
        lambda_ = regression.params[0]                     # (viii) Get Coefficient (λ)
        if lambda_ >= 0:                                   # (ix)   If λ is not negative, spread does not mean revert, return none
            return None
        else:
            return -np.log(2) / lambda_                    # (x)    If λ is negative return the half life
    
    '''
        get_zero_line_crossings - takes in the spread and returns number of times it has 
        crossed the 0 line.
    '''
    def get_zero_line_crossings(self, spread):
        return len(np.where(np.diff(np.sign(spread)))[0])

    '''
        get_pair_stats: returns a dictionary for the pair created by the two symbols passed 
        that contians the pair's ADF value, p-value, significance level, half life duration and 
        0-line crossings.
    '''
    def get_pair_stats(self, symbol_1 : str, symbol_2 : str, start_date : str, end_date : str):
        # (1) Initialize DATA OBJ if not initialized
        if self.DATA.is_initialized() == False:
            self.DATA.initialize_data()

        # (2) Get symbols' data & their closing prices 
        symbol_1_data = self.DATA.get_data_between_dates(symbol_1, start_date, end_date)
        symbol_2_data = self.DATA.get_data_between_dates(symbol_2, start_date, end_date)
        close_prices_1 = symbol_1_data['Close']
        close_prices_2 = symbol_2_data['Close']

        # (3) Calculate hedge ratio & spread of symbols
        hedge_ratio = self.DATA.calculate_hedge_ratio(close_prices_1, close_prices_2)
        spread = self.DATA.calculate_spread(close_prices_1, close_prices_2, hedge_ratio)

        # (4) Create and return pair's stats dictionary
        adf_result = adfuller(spread)
        adf_value  = adf_result[0]
        p_value    = adf_result[1]
        sig_level  = self.get_hypothesis_level(p_value, adf_value)
        pair_stats = {'Pair'                              : f'({symbol_1}-{symbol_2})',
                      'ADF Value'                         : round(adf_value, 5),
                      'P-Value'                           : round(p_value, 5),
                      'Mean Reversion Significance Level' : sig_level,
                      'Half Life'                         : self.get_half_life(spread = spread),
                      '0-line Crossings'                  : self.get_zero_line_crossings(spread = spread)
                     }
        return pair_stats
    
    ''' 
        get_pair_and_strength_from_str: helper function to parse string entry of union-intersection-difference df
            => output 1: pair     : string
            => output 2: strength : float
            => output 3: symbols  : list of length 2
            => ex: "('DHI-LEN', 54.0)" => 'DHI-LEN', 54.0, ['DHI', 'LEN']
    '''
    def get_pair_and_strength_from_str(self, pair_entry : str):
        # (1) Remove parentheses and split the entry by comma
        pair_and_strengths = pair_entry.strip("()").split(",") # could be ('A-B', 20, 30) or ('A-B', 30)
        pair = pair_and_strengths[0]                           # strip spaces and quotes from pair 
        pair = pair.strip(" '\"")

        # (2) Build entry for strength (depending of if we have one or two values)
        strength = str(pair_and_strengths[1]) if len(pair_and_strengths) == 2 else f'({pair_and_strengths[1][1:]}, {pair_and_strengths[2]})'
        
        # Split the pair into symbols
        symbols = pair.split("-")
        
        return pair, strength, symbols

    '''
    get_all_pairs_stats: returns dataframe with pairs & their statistical arbitrage values
        => inputs:
            (1) path_to_pairs_csv: Path to csv file with pairs in the union, intersection & difference
            (2) start_date , end_date: start & end dates for the data to be used to calculate statistical arbitrage values

        => output: dataframe like such
            | Pair | Correlational Strength | Half Life | ADF Value | P-Value | Mean Reversion Significance Level | Group |
    '''
    def get_all_pairs_stats(self, path_to_pairs_csv : str, start_date : str, end_date : str):
        # (1) Initialize DATA OBJ if not initialized
        if self.DATA.is_initialized() == False:
            self.DATA.initialize_data()

        # (2) Initialize union-intersection-differences file
        pairs_df = pd.read_csv(path_to_pairs_csv)

        # (3) Initialize output dataframe
        all_pairs_stats = pd.DataFrame()

        # (4) Iterate through all columns in pairs data frame
        visited_pairs = []
        failed_pairs  = []
        next_milestone = 0
        curr_pairs_completed = 0

        for col in ['Intersection', 'File 1 Exclusive Pairs', 'File 2 Exclusive Pairs', 'Union']:
            pairs_in_col =  pairs_df[col].tolist() 

            for pair_entry in pairs_in_col:
                 # (1) Filter out NaNs
                if isinstance(pair_entry, str) == False:   
                    continue
            
                # (2) Parse pair_entry string 
                pair, correlational_strength, symbols = self.get_pair_and_strength_from_str(pair_entry)

                # (3) Skip if pair has been visited before (i.e pair was in another column)
                if pair in visited_pairs:
                    continue 
                else: 
                    visited_pairs.append(pair)
                
                # (4) Get pair'stats with class method 
                try:
                    curr_pair_stats = self.get_pair_stats(symbol_1 = symbols[0], symbol_2 = symbols[1], 
                                                          start_date = start_date, end_date = end_date)
                except:
                    failed_pairs.append(pair)
                    continue 

                # (5) Augment current pair's stats with correlation strength and their group 
                curr_pair_stats['Correlational Strength'] = correlational_strength
                curr_pair_stats['Group'] = col if col == 'union' else f'{col}, union'

                # (6) Create new row for current pair's stats in output dataframe 
                curr_pair_row = pd.DataFrame(curr_pair_stats, index = [0])
                if all_pairs_stats.empty:
                    all_pairs_stats = curr_pair_row
                else:
                    all_pairs_stats = pd.concat([all_pairs_stats, curr_pair_row], ignore_index = True)
                
                # (7) Update progress and print
                curr_pairs_completed += 1
                if curr_pairs_completed >= next_milestone:
                    print(f" >> COMPLETED A TOTAL OF {curr_pairs_completed} PAIRS")
                    next_milestone += 500

        # Drop any columns with any nones
        return all_pairs_stats, failed_pairs
    
    '''
        rank_pairs: Ranks pairs from best to worst using specified weights 
        for each statistical metric. 
            Input:
                => weights_per_col - dictionary where keys = metric and values = importance weight for that metric 
                => start_date, end_date - strings to determine time period to get the best pairs 
                => path_to_pairs_csv - dataframe / csv that has all pairs we need to assess 
            
            Output: 
                => dataframe | best pair at the top 

    '''
    def rank_pairs(self, path_to_pairs_csv : str, start_date : str, end_date : str, 
                   weights_per_col : dict = {"ADF Value" : 1, "P-Value" : 1, "Half Life" : 1, 
                                            "Mean Reversion Significance Level" : 1, '0-line Crossings' : 1}):
        # (0) Get all pairs stats w/class helper method
        all_pairs_stats, failed = self.get_all_pairs_stats(path_to_pairs_csv, start_date, end_date)

        # (1) Prepare dataframe for min-max scaling
        min_max_scaler = MinMaxScaler()
        all_pair_stats_min_max = all_pairs_stats.copy()
        all_pair_stats_min_max.drop(["Pair", "Group", "Correlational Strength"], axis = 1, inplace = True)

        print(f'Could not retrieve information of {len(failed)} pairs: \n {failed}')
        print(all_pairs_stats.columns)

        # (2) Apply transformations to columns where the smallest value == the best
        def transform(x):
            x = -x if x > 0 else abs(x)
            return x

        for col in ["ADF Value", "P-Value", "Half Life"]:
            all_pair_stats_min_max[col] = all_pair_stats_min_max[col].apply(lambda x: transform(x))

        # (3) Apply min-max scalar to columns
        scaled_array  = min_max_scaler.fit_transform(all_pair_stats_min_max)
        all_pair_stats_min_max = pd.DataFrame(scaled_array, columns = all_pair_stats_min_max.columns)

        # (4) Apply weight multiplier to min-max scaled columns
        for column in ['ADF Value', 'P-Value', 'Half Life', 'Mean Reversion Significance Level', '0-line Crossings']:
            try: 
                weight_for_column = weights_per_col[column]
                all_pair_stats_min_max[column] = all_pair_stats_min_max[column] * weight_for_column
            except:
                all_pair_stats_min_max[column] = 0

        # (5) Create ranking metric column for all rows
        all_pair_stats_min_max["Ranking Score"] = all_pair_stats_min_max.sum(axis = 1)

        # (6) Augment non-scaled  "all-pair_stats" data frame with ranking metric column
        non_scaled = all_pairs_stats.copy()
        non_scaled["Ranking Score"] = all_pair_stats_min_max["Ranking Score"]

        # (7) Sort all-pairs-stats dataframe by ranking metric value (highest ranking score values @ top)
        non_scaled = non_scaled.sort_values(by = "Ranking Score", ascending = False)

        # (8) Reset Index of non-scaled copy
        non_scaled.reset_index(drop = True, inplace = True)

        return non_scaled
    
    '''
        get_top_pairs_from_group - This function takes three inputs: a sorted ranked dataframe, 
        the number of pairs to retrieve, and a group name. It returns the specified number of 
        top pairs from the given group. 
            ** group name = 'Union', 'Intersection', 'File (1/2) Exclusive Pairs' ** 
            ** if pairs_to_retrieve == None, it returns all pairs from that group ** 
    '''
    def get_top_pairs_from_group(self, ranking_df, pairs_to_retrieve : int, group_name : str):
        if group_name == 'Union':
            group_df = ranking_df[ranking_df['Group'].str.contains('union', case=False)]
        else:
            group_df = ranking_df[ranking_df['Group'].str.contains(group_name, case=False)]
        
        final = group_df if pairs_to_retrieve == None else group_df.head(pairs_to_retrieve)
        
        return final 
    
    '''
        group_statistics - This function takes a sorted ranked dataframe and a significance level, 
        and returns the mean values of various statistics for each group.

        The returned statistics for each group include:
        - Number of pairs within the specified significance level
        - Percentage of pairs within the specified significance level
        - Average 0-Crossings for pairs within the significance level
        - Average ADF (Augmented Dickey-Fuller test statistic) for pairs within the significance level
        - Average P-Value for pairs within the significance level
    '''
    def group_stastics(self, ranking_df, level : int = 3):
        return_df = pd.DataFrame()
        for group in ['Union', 'Intersection', 'File 1 Exclusive Pairs', 'File 2 Exclusive Pairs']:
            new_row : dict = dict()

            # (i) Get pairs in group with passed in significance level 
            group_df = self.get_top_pairs_from_group(ranking_df = ranking_df, pairs_to_retrieve = None, group_name = group)
            group_df_level = group_df[group_df['Mean Reversion Significance Level'] == level] 

            # (ii) Populate group's row dictionary 
            new_row['Group'] = group
            new_row[f'Pairs in Significance Level {level}'] = group_df_level.shape[0]
            new_row[f'Percentage of Pairs in Significance Level {level}'] = round((group_df_level.shape[0] / group_df.shape[0])* 100, 2)
            new_row[f'Average ADF Value for Level {level} pairs'] = round(group_df_level['ADF Value'].mean(), 2)
            new_row[f'Average P-Value for Level {level} pairs'] = round(group_df_level['P-Value'].mean(), 2)
            new_row[f'Average 0-Crossings for Level {level} pairs'] = round(group_df_level['0-line Crossings'].mean(), 5)
            new_row = pd.DataFrame([new_row])

            # (iii) Add group's row to data frame we will return 
            return_df = pd.concat([return_df, new_row], axis = 0) if return_df.empty == False else new_row
        
        return return_df



