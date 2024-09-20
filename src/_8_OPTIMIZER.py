from _0_helper_functions import *
from _5_DATA_MANIPULATOR import DATA_MANIPULATOR
from _6_ZSCORE_STRATEGY  import ZSCORE_STRATEGY
from _7_PAIRS_BACKTESTER import PAIRS_BACKTESTER

"""
                                    << OPTIMIZER (module) >>
    => Class Initialization Inputs:
        (1) pairs_list                    : list  | list of pairs to optimize
        (2) start_date                    : str   | start date for the optimization period
        (3) end_date                      : str   | end date for the optimization period
        (4) path_to_prices_df             : str   | path to the DataFrame containing price data
        (5) strategy_name                 : str   | name of the strategy to optimize
        (6) parameters_ranges             : list  | list of parameter ranges for optimization
        (7) initial_capital_per_backtest  : float | initial capital for each backtest
        (8) fixed_allocation_per_backtest : float | fixed allocation for each trade
        (9) risk_per_trade                : float | risk percentage per trade

    ------------ User-facing Methods --------------
        (1) optimize(results_to_optimize, results_to_show=10, show_results=False):
            => Optimizes the backtest results based on user-specified evaluation criteria.
            => Inputs:
                results_to_optimize : dict | dictionary of evaluation metrics and their weights
                results_to_show     : int  | number of top results to display
                show_results        : bool | whether to display the results
            => Outputs:
                DataFrame | top optimized results

        (2) get_average_of_all_results(method="ALL PAIRS"):
            => Returns the average value of all metrics across all pairs or individual pairs.
            => Inputs:
                method : str | method to calculate the average ("ALL PAIRS" or "SINGLE PAIR")
            => Outputs:
                Series or dict | average values of metrics

    ------------ Backend Methods --------------
        (1) get_combinations():
            => Filters and returns parameter combinations that comply with the strategy's rules.
            => Outputs:
                list | filtered list of parameter combinations

        (2) get_combinations_helper():
            => Generates all possible combinations of parameters based on the specified ranges.
            => Outputs:
                list | list of parameter combinations

        (3) run_zscore_backtests_single_pair(symbol_1, symbol_2, all_param_combinations):
            => Runs all parameter combinations for a single pair.
            => Inputs:
                symbol_1             : str  | first symbol of the pair
                symbol_2             : str  | second symbol of the pair
                all_param_combinations : list | list of parameter combinations
            => Outputs:
                DataFrame | results of backtests for the single pair

        (4) run_zscore_backtests_all_pairs():
            => Runs the optimizer for all pairs and stores the results.
            => Outputs:
                None

        (5) apply_min_max_to_backtests():
            => Applies min-max scaling to the backtest results.
            => Outputs:
                None
"""

class OPTIMIZER():
    def __init__(self, pairs_list : list, start_date : str, end_date : str, path_to_prices_df : str,
                 strategy_name : str, parameters_ranges : list , initial_capital_per_backtest : float, 
                 fixed_allocation_per_backtest : float, risk_per_trade : float):
        # (1) Initialize Strategy OBJ Parameters
        self.pairs_list         = pairs_list
        self.start_date         = start_date
        self.end_date           = end_date
        self.path_to_prices_df  = path_to_prices_df
        self.parameters_ranges  = parameters_ranges
        self.strategy_name      = strategy_name

        # (2) Initialize Backtesting OBJ Parameters
        self.initial_capital  = initial_capital_per_backtest
        self.fixed_allocation = fixed_allocation_per_backtest
        self.risk_per_trade   = risk_per_trade

        # (3) Initialize Output Variabels
        self.backtests_init = False
        self.all_backtests_df = pd.DataFrame()
        self.all_backtests_min_max_scaled = pd.DataFrame()
    
    '''
        get_combinations_helper: takes the cross product of all parameter ranges  
        get_combinations: returns all combinations that comply with each strategy's rules   
    '''
    def get_combinations_helper(self):
        range_values = [np.arange(start, end + step, step) for start, end, step in self.parameters_ranges]
        combinations = []
        combinations = list(product(*range_values))
        return combinations

    def get_combinations(self):
        # (1) Create all possible combiniations of parameters using user defined ranges
        all_param_combinations = self.get_combinations_helper()
        if all_param_combinations == [] or all_param_combinations == [()]:
            print("No Combinations Found")
            return

        # (2) Filter combinations where abs(ZSCORE ENTRY THRESHOLD) > abs(Z STOP LOSS) if we are using ZSCORE STRATEGY
        if self.strategy_name == "ZSCORE_STRATEGY":
            all_param_combinations = [i for i in all_param_combinations if abs(i[3]) <= abs(i[1])]

        # (3) print out number of combinations and return them
        print(f"Combinations Requested per Pair: {len(all_param_combinations)} | Total Pairs: {len(self.pairs_list)} | Total Iterations: {len(all_param_combinations) * len(self.pairs_list)} \n")
        return all_param_combinations
    
    '''
        run_zscore_backtests_single_pair: runs optimizer for one single pair (i.e runs all combinations of parameters for one pair)

        run_zscore_backtests_all_pairs: uses 'run_zscore_backtests_single_pair' on all pairs of class attributre 'pairs_list'. It
        stores all results in class attribute 'all_backtests_df'.

    '''
    def run_zscore_backtests_single_pair(self, symbol_1 : str, symbol_2 : str, all_param_combinations : list):
        # (1) Initialize progress tracking variables
        num_total_combinations = len(all_param_combinations)
        combinations_done = 0
        next_completeness_milestone = 0 # <- percent

        # (2) Initialize output variable 
        all_backtests = pd.DataFrame()

        # (3) Run all combinations 
        while all_param_combinations != []:
            curr_combo = all_param_combinations[0]
            all_param_combinations = all_param_combinations[1:]

            # (i) if its first combination, initialize strategy obj
            if combinations_done == 0:
                strategy_obj = ZSCORE_STRATEGY( symbol_1           = symbol_1,
                                                symbol_2           = symbol_2,
                                                path_to_prices_df  = self.path_to_prices_df,
                                                start_date         = self.start_date,
                                                end_date           = self.end_date,
                                                zscore_period      = curr_combo[0],
                                                zscore_entry       = curr_combo[1],     
                                                zscore_stop_loss   = curr_combo[2], 
                                                zscore_take_profit = curr_combo[3],
                                                use_coint_check    = curr_combo[4],
                                                coint_check_length = curr_combo[5],
                                                use_days_stop      = curr_combo[6],
                                                days_for_stop      = curr_combo[7]
                                              )
            # else, simply change the paramters of existing object
            else:
                strategy_obj.change_parameters( symbol_1           = symbol_1, 
                                                symbol_2           = symbol_2, 
                                                zscore_period      = curr_combo[0],
                                                zscore_entry       = curr_combo[1],     
                                                zscore_stop_loss   = curr_combo[2], 
                                                zscore_take_profit = curr_combo[3],
                                                use_coint_check    = curr_combo[4],
                                                coint_check_length = curr_combo[5],
                                                use_days_stop      = curr_combo[6],
                                                days_for_stop      = curr_combo[7]
                                              )
            # (ii) Initialize backtester obj and run backtest with current combination of parameters
            backtester = PAIRS_BACKTESTER(strategy_obj = strategy_obj, initial_capital = self.initial_capital,
                                          fixed_allocation = self.fixed_allocation, risk_per_trade = self.risk_per_trade)   
            curr_backtest_results = backtester.backtest()
            
            # (iii) Append results of current backtest iteration to 'all_backtests'
            curr_backtest_results['Pair'] = f"({symbol_1}-{symbol_2})"
            curr_backtest_results['zscore_period']      = curr_combo[0],
            curr_backtest_results['zscore_entry']       = curr_combo[1],     
            curr_backtest_results['zscore_stop_loss']   = curr_combo[2], 
            curr_backtest_results['zscore_take_profit'] = curr_combo[3],
            curr_backtest_results['use_coint_check']    = curr_combo[4],
            curr_backtest_results['coint_check_length'] = curr_combo[5],
            curr_backtest_results['use_days_stop']      = curr_combo[6],
            curr_backtest_results['days_for_stop']      = curr_combo[7]
            if combinations_done == 0:
                all_backtests = pd.DataFrame(curr_backtest_results)
            else:
                all_backtests = pd.concat([all_backtests, pd.DataFrame(curr_backtest_results, index = [0])], ignore_index = True)
            
            # (iv) Print progress 
            combinations_done += 1
            complete_percentage = round(combinations_done / num_total_combinations, 3) * 100
            if complete_percentage > next_completeness_milestone:
                next_completeness_milestone += 25
                pair_string         = f"({symbol_1}-{symbol_2})"
                combinations_string = f"{combinations_done}/{num_total_combinations}"
                percentage_string   = f"{round(complete_percentage, 2)}%"
                len_for_combo       = len(str(num_total_combinations)) * 2 + 1
                print(f"Current Pair: {pair_string:<10} | Completed Combinations: {combinations_string:<10} | Current Progress: {percentage_string:<5}")

        return all_backtests

    def run_zscore_backtests_all_pairs(self):
        all_combinations = self.get_combinations()
        pairs_completed = 1
        total_pairs = len(self.pairs_list)
        for pair in self.pairs_list:
            # (i) Run all backtests of pair w/'run_zscore_backtests_single_pair'
            all_backtests_for_curr_pair = self.run_zscore_backtests_single_pair(pair[0], pair[1], all_combinations)
            
            # (ii) Concatenate current pair results with results of other pairs
            if self.all_backtests_df.empty:
                self.all_backtests_df = all_backtests_for_curr_pair
            else:
                self.all_backtests_df = pd.concat([self.all_backtests_df, all_backtests_for_curr_pair], ignore_index = True)
            
            # (iii) Print Progress Report
            print("\n")
            print(f" >> TOTAL PAIRS COMPLETED: {pairs_completed}/{total_pairs} | CURRENT PROGRESS: {round(pairs_completed / total_pairs, 3) * 100}%")
            print("\n")
            pairs_completed += 1
        
        # Show 'Pair' column as the first column of 'all_backtests_df'
        self.all_backtests_df = self.all_backtests_df.reindex(columns = ["Pair"] + [col for col in self.all_backtests_df.columns if col != "Pair"])

        # Drop any rows (i.e backtests) that have 'None' in any of their columns 
        self.all_backtests_df = self.all_backtests_df.dropna()

        # Record that backtests have been ran 
        self.backtests_init = True 
        return
    
    '''
        apply_min_max_to_backtests: applies a min-max scalar on all result columns of the class attribute 'all_bactests_df'. It stores 
        the results in another class attribute - 'all_backtests_min_max'
    '''
    def apply_min_max_to_backtests(self):
        # (1) Initialize min-max scaler 
        scalar = MinMaxScaler()

        # (2) Copy 'all_backtests_df' into min-max class attribute 
        self.all_backtests_min_max_scaled = self.all_backtests_df.copy()

        # (3) Filter out parameter columns that should not be normalized
        non_normalizeable = ["zscore_value", "zscore_period", "zscore_stop_loss", "zscore_take_profit", "use_coint_check", "coint_check_length", "Pair"]
        columns_to_normalize = [col for col in self.all_backtests_df.columns if col not in non_normalizeable]

        # (4) Apply min-max scaler to the correct columns
        self.all_backtests_min_max_scaled[columns_to_normalize] = scalar.fit_transform(self.all_backtests_df[columns_to_normalize])


    '''
        optimize: Runs backtests and creates an optimization metric based on user-specified evaluation criteria.
                  Returns a sorted DataFrame based on this optimization metric.

            Input:
                => results_to_optimize: dict 
                    A dictionary where:
                        - Keys are evaluation metrics (e.g., 'Sharpe Ratio', 'Win Rate')
                        - Values are weights for these metrics
                    Example: {'Sharpe Ratio': 1, 'Win Rate': 2}
                    Any metrics not specified in this dictionary are assigned a default weight of 0.
                
                
                => results_to_show : int 
                    How many of the top results would the user like to show
            
            Output:
                => dataframe | the best few results given the metrics user wanted to optimize
    '''
    def optimize(self, results_to_optimize : dict, results_to_show : int = 10, show_results : bool = False):
        # (1) Run backtests of all pairs (if it has not been done)
        if self.backtests_init == False:
            if self.strategy_name == "ZSCORE_STRATEGY":
                self.run_zscore_backtests_all_pairs()
                self.apply_min_max_to_backtests()
        
        # (2) Create simplied 'all_backtests_min_max_scaled' where we only keep columns in 'results_to_optimize'
        simple_min_max = pd.DataFrame()
        for column_key in results_to_optimize:
            weight = results_to_optimize[column_key] if column_key != 'Drawdown' else -results_to_optimize[column_key]
            simple_min_max[column_key] = self.all_backtests_min_max_scaled[column_key].tolist()
            simple_min_max[column_key] = simple_min_max[column_key] * weight # apply the weight
        
        # (3) Create optimization metric column for 'simple_min_max'
        simple_min_max['Optimization Metric'] = simple_min_max.sum(axis = 1)

        # (4) Augment 'all_backtests_df' with the optimization metric column 
        copy_all_backtests_df = self.all_backtests_df.copy()
        copy_all_backtests_df['Optimization Metric'] = simple_min_max['Optimization Metric'].tolist()

        # (5) Sort by optimization metric and return top results requested 
        copy_all_backtests_df = copy_all_backtests_df.sort_values(by = "Optimization Metric", ascending = False)    
        copy_all_backtests_df = copy_all_backtests_df.head(results_to_show)
        copy_all_backtests_df = copy_all_backtests_df.reset_index()
        copy_all_backtests_df = copy_all_backtests_df.drop(columns = ['index'], inplace = False)

        if show_results == True:
            print(copy_all_backtests_df.to_string())

        return copy_all_backtests_df
    
    '''
        get_average_of_all_results : returns the mean value of all metrics across either all pairs 
        or per individual pair 
    '''
    def get_average_of_all_results(self, method = "ALL PAIRS"):
        if self.backtests_init == False:
            if self.strategy_name == "ZSCORE_STRATEGY":
                self.run_zscore_backtests()

        if method == "ALL PAIRS":
            numericals = self.all_backtests_df.select_dtypes(include=['number'])
            return numericals.mean()

        elif method == "SINGLE PAIR":
            # (i) Build Dictionary Where Key-> Pair, Value -> pair's results
            results_dict = dict()
            for pair in self.pairs_list:
                curr_pair_backtests = self.all_backtests_df[self.all_backtests_df["Pair"] == f"({pair[0]}-{pair[1]})"]
                curr_pair_backtests = curr_pair_backtests.drop(columns = ["Pair"])
                curr_pair_results   = curr_pair_backtests.mean()
                results_dict[f"({pair[0]}-{pair[1]})"] = curr_pair_results

            for key in results_dict:
                print(f"PAIR: {key}")
                print(results_dict[key])
                print("\n")
        
        return results_dict
    