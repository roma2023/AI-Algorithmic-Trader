from _0_helper_functions import *
'''
                                            << PAIR_FINDER (module) >>
    => Class Initialization Inputs:
        (1) voting_rule : str  | rule to be used for voting (e.g., 'Borda Count', 'Feature by Feature')
        (2) number_of_pairs_per_ticker : int  | number of pairs to be identified per ticker

    ------------ User-facing Methods --------------
        (1) create_corr_matrix(path_to_prices_or_sent : str, path_to_save : str = None):
            => Creates and saves a correlation matrix from the given data.
            => Inputs:
                - path_to_prices_or_sent: str  | path to the file with prices or sentiment data
                - path_to_save: str            | path to save the correlation matrix (optional)
            => Output:
                - dataframe | correlation matrix of the data.

        (2) get_top_pairs_of_all_tickers(corr_matrix):
            => Returns the top correlated pairs for all tickers based on the voting rule.
            => Inputs:
                - corr_matrix: dataframe  | correlation matrix of the data
            => Output:
                - dataframe | tickers and their top correlated pairs.

    ------------ Backend Methods --------------
        (1) transponse_ticker(df, ticker : str, name_of_date_column : str = 'date'):
            => Returns the transposed data of the specified ticker.
            => Inputs:
                - df: dataframe              | original data
                - ticker: str                | ticker symbol
                - name_of_date_column: str   | name of the date column
            => Output:
                - dataframe | transposed data of the ticker.

        (2) transpose_all_tickers(df, name_of_date_column : str = 'date'):
            => Returns the transposed data of all tickers.
            => Inputs:
                - df: dataframe              | original data
                - name_of_date_column: str   | name of the date column
            => Output:
                - dataframe | transposed data of all tickers.

        (3) get_top_correlations_for_feature(corr_matrix, feature : str, top_X : int = 10):
            => Returns the top correlated features for the specified feature.
            => Inputs:
                - corr_matrix: dataframe  | correlation matrix
                - feature: str            | feature name
                - top_X: int              | number of top correlations to return
            => Output:
                - dataframe | top correlated features for the specified feature.

        (4) get_top_correlations_for_all_features(corr_matrix, top_X : int = 10):
            => Returns the top correlated features for all features.
            => Inputs:
                - corr_matrix: dataframe  | correlation matrix
                - top_X: int              | number of top correlations to return
            => Output:
                - dataframe | top correlated features for all features.

        (5) borda_count_on_ticker(top_df, top_X : int, ticker : str, all_tickers : list):
            => Returns the most correlated tickers for a specific ticker using Borda Count.
            => Inputs:
                - top_df: dataframe      | top correlated features for all features
                - top_X: int             | number of top correlations to return
                - ticker: str            | ticker symbol
                - all_tickers: list      | list of all ticker symbols
            => Output:
                - dataframe | most correlated tickers for the specified ticker.

        (6) borda_count_on_all_tickers(top_df, top_X : int, all_tickers : list):
            => Returns the most correlated tickers for all tickers using Borda Count.
            => Inputs:
                - top_df: dataframe      | top correlated features for all features
                - top_X: int             | number of top correlations to return
                - all_tickers: list      | list of all ticker symbols
            => Output:
                - dataframe | most correlated tickers for all tickers.

        (7) feature_by_feature_on_ticker(corr_matrix, ticker : str, top : int, tickers : list):
            => Returns the most correlated tickers for a specific ticker using Feature by Feature method.
            => Inputs:
                - corr_matrix: dataframe  | correlation matrix
                - ticker: str             | ticker symbol
                - top: int                | number of top correlations to return
                - tickers: list           | list of all ticker symbols
            => Output:
                - dataframe | most correlated tickers for the specified ticker.

        (8) feature_by_feature_on_all_tickers(corr_matrix, top : int, tickers : list):
            => Returns the most correlated tickers for all tickers using Feature by Feature method.
            => Inputs:
                - corr_matrix: dataframe  | correlation matrix
                - top: int                | number of top correlations to return
                - tickers: list           | list of all ticker symbols
            => Output:
                - dataframe | most correlated tickers for all tickers.
'''
class PAIR_FINDER():
    def __init__(self, voting_rule : str, number_of_pairs_per_ticker : int = 10): 
        # (1) Initilize parameters into class 
        self.voting_rule                = voting_rule
        self.number_of_pairs_per_ticker =  number_of_pairs_per_ticker

    @staticmethod
    def multi_core(args):
        obj, corr_matrix, ticker, top, tickers, method = args
        if method == 'Feature by Feature':
            return obj.feature_by_feature_on_ticker(corr_matrix, ticker, top, tickers)
    
    '''
        create_corr_matrix - creates correlation matrix from data passed in, saves it 
        to specified path and returns it.
    '''
    def create_corr_matrix(self, path_to_prices_or_sent : str, path_to_save : str = None):
        prices_or_sent_data = pd.read_csv(path_to_prices_or_sent)         # (i)   Initialize Sentimient/Quantitative data
        transposed = self.transpose_all_tickers(df = prices_or_sent_data) # (ii)  Transpose sentiment/quantiative data 
        transposed = transposed.drop(columns = ['date'], inplace = False) # (iii) Drop date column 
        corr_matrix = transposed.corr()                                   # (iv)  Create correlation matrix 
        if path_to_save != None:
            corr_matrix.to_csv(path_to_save, index = False)                              # (v)   Save correlation matrix to specified path
        return corr_matrix
    '''
        get_top_pairs_of_all_tickers - gets the 'number_of_pairs_per_ticker' requested by 
        user when creating 'PAIR_FINDER' obj  for all pairs and returns it in a dataframe 
        like such: 
            | Ticker |     Top 1   |     Top 2    | ... | Top X 
                A      ('ABT', 20)   ('AMD', 10) ... 
                B      ...
    '''
    def get_top_pairs_of_all_tickers(self, corr_matrix) : 
        # Get tickers from corr_matrix 
        tickers = get_tickers_from_corr_matrix(corr_matrix = corr_matrix)
        
        # Case 1 - Finding Pairs with Borda Count 
        if self.voting_rule == 'Borda Count': 
            features_top = self.get_top_correlations_for_all_features(corr_matrix = corr_matrix, top_X = self.number_of_pairs_per_ticker)  
            pairs_df = self.borda_count_on_all_tickers(top_df = features_top, top_X = self.number_of_pairs_per_ticker, all_tickers = tickers) 
        
        # Case 2 - Finding Pairs with Feature by Feature with multiple cores
        else: 
            num_processes = max(1, cpu_count() // 4)
            print(f'Using {num_processes} out of {cpu_count()} CPUs')   
            with Pool(processes = num_processes) as pool:
                results = pool.map(PAIR_FINDER.multi_core, [(self, corr_matrix, ticker, self.number_of_pairs_per_ticker, tickers, self.voting_rule) for ticker in tickers])
            
            pairs_df = pd.concat(results, ignore_index=True)

        # (3) Return voting results 
        pairs_df.reset_index(drop = True, inplace = True)
        return pairs_df              
    
    '''
        transpose_ticker - returns a data frame of the ticker's data columns transposed like such: 
        |   date     | ticker |  i_1  |  i_2  |            |   date    | A_i_1 | A_i_2 |
        '2018-01-01'    A        2.1     2.0               '2018-01-01'   2.1     2.0
        '2018-01-02'    A        2.4     3.8               '2018-01-02'   2.4     3.8
        ⋮                                           ====>
        '2018-01-01'    B       10.1     9.1
        '2018-01-02'    B       8.1      7.0

        transpose_all_tickers - returns a data frame of all ticker's data transposed (not just an 
        individual ticker like above)
        |   date     | ticker |  i_1  |  i_2  |            |   date    | A_i_1 | A_i_2 | B_i_1 | B_i_2 |
        '2018-01-01'    A        2.1     2.0               '2018-01-01'   2.1     2.0     10.1    9.1
        '2018-01-02'    A        2.4     3.8               '2018-01-02'   2.4     3.8     8.1     7.0
        ⋮                                           ====>   ⋮ 
        '2018-01-01'    B       10.1     9.1
        '2018-01-02'    B       8.1      7.0
    '''
    def transponse_ticker(self, df, ticker : str, name_of_date_column : str = 'date'):
        ticker_data = getSymbolData(df = df, symbol = ticker)   # (i)   Get individual ticker data 
        ticker_data = ticker_data.drop(columns = ['ticker'])    # (ii)  Drop the ticker column 
        for column in ticker_data.columns:                      # (iii) Add ticker name as prefix to columns 
            if column != name_of_date_column: 
                ticker_data.rename(columns = {column: f'{ticker}_' + column}, inplace = True)
        return ticker_data
    
    def transpose_all_tickers(self, df, name_of_date_column :str = 'date'):
        all_tickers : list = getTickers(df)
        merged_df = pd.DataFrame()
        while all_tickers != []:
            curr_ticker_transposed = self.transponse_ticker(df = df, ticker = all_tickers[0], name_of_date_column = name_of_date_column)
            merged_df = mergeDataFrames(df_1 = merged_df, df_2 = curr_ticker_transposed, column_to_merge = name_of_date_column)
            all_tickers = all_tickers[1:]
        return merged_df
    
    '''
        get_top_correlations_for_feature - returns the 'top_X' most correlated features to a specific feature in a correlation matrix in a
        dataframe in the following format:

        Format of all cells  in a 'Top_i' column:
                "[Indicator Name] / [Correlation value of that indicator in relation the feature passed into func]" : string 
            
        Format of returned data frame: 
        => (ex) get_top_correlations(some_matrix, 'A_RSI_14')
                | Indictator  |                Top_1               | Top_2 | ... | Top_X
                | A_RSI_14    |  [TMO_CCI_40_c= 0.015] / [0.647] | ... 

        get_top_correlations_for_all_features - returns a dataframe with the 'top_X' most correlated features to 
        all features in a correlation matrix in a dataframe in the following format:

        Format of all cells  in a 'Top_i' column:
                "[Indicator Name] / [Correlation value of that indicator in relation the feature passed into func]" : string 
        
        Format of returned data frame:
         | Indictator  |                Top_1               | Top_2 | ... | Top_X
         | A_RSI_14    |  [TMO_CCI_40_c= 0.015] / [0.647]   | ... 
         | A_MACD      |  [MSFT_MACD] / [0.90]              | ... 
             ⋮                          ⋮
    '''
    def get_top_correlations_for_feature(self, corr_matrix, feature : str, top_X : int = 10):
        ticker    : str  = feature.split('_')[0]  # (1) Get respective ticker of feature (ex: 'AMD_RSI_14' -> ['AMD','RSI', 14])
        corr_dict : dict = corr_matrix[feature]   # (2) Keys = all other features in 'corr_matrix', values = corr value in relation to 'feature'

        # (3) Ensure values in corr_dict are numeric & sort dictionary by absolute values
        corr_dict : dict = {k: float(v) for k, v in corr_dict.items() if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())} # (iii) Ensure values in corr_dict are numeric 
        sorted_corr_dict = sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        # (4) Filter out features with same ticker prefix as our 'feature'
        filtered_sorted_corr_dict = dict()
        for key, value in sorted_corr_dict:
            if isinstance(key, str) and key.split('_')[0] != ticker:
                filtered_sorted_corr_dict[key] = value
        
        # (5) Get a dictionary of only the top 'number_of_pairs_per_ticker'
        top : dict = dict(list(filtered_sorted_corr_dict.items())[:top_X])

        # (6) Build return data frame 
        top_correlations_for_feature = {'Indicator' : feature}
        i = 1
        for key, value in top.items():
            curr_top_num = f'Top_{i}'
            top_correlations_for_feature[curr_top_num] = f'[{key}] / [{value}]'
            i += 1

        return pd.DataFrame(top_correlations_for_feature, index = [0])

    def get_top_correlations_for_all_features(self, corr_matrix, top_X : int = 10):
        top_df = pd.DataFrame()                     # (1) Initialize output data frame
        all_features : list = corr_matrix.columns   # (2) Get list of all features in corr_matrix (which is just it's columns)
        
        # (3) Iterate through all features, get their closest correlated features & concatenate their results to a dataframe 
        for feature in all_features:
            curr_feature_top_corr = self.get_top_correlations_for_feature(corr_matrix = corr_matrix, feature = feature, top_X = top_X)
            top_df = curr_feature_top_corr if top_df.empty else pd.concat([top_df, curr_feature_top_corr], ignore_index = True)

        return top_df   

    '''
        borda_count_on_ticker - returns most correlated tickers of a specific ticker based on 
        the features that have the passed ticker as a prefix (in 'top_df'). 
        (ex)
        'top_df' = 
            | Indictator |                Top_1               | Top_2 | ... | Top_X
            | A          |  [TMO_CCI_40_c= 0.015] / [0.647]   | ... 
            | A          |  [MSFT_MACD] / [0.90]              | ... 
            | B          |  [AAPL_CLOSE] / [0.99]             | ...
                ⋮  
        
        borda_count_on_ticker(top_df, top_X = 2, 'A') ==> 
            | Ticker  |    Top_1    |    Top_2   ...
            |    A    | (MSFT, 20)  |  (TMO, 10) ...

        * Note: max votes for any ticker = num(features in corr matrix) * Σ(1 - top_X)
        because each feature of a ticker awards X votes to 1st then X-1 to 2nd, etc. Thus 
        it could be the case that all features had all X other features of a particular ticker.
            => (ex) max  = 93 * (10+9+...+1) = 93 * 55 = 5115
        ---------------------------------------------------------------------------------------
        borda_count_on_all_tickers - returns the most correlated tickers for all tickers.  

        'top_df' = 
            | Indictator |                Top_1               | Top_2 | ... | Top_X
            | A_OPEN     |  [TMO_CCI_40_c= 0.015] / [0.647]   | ... 
            | A_CLOSE    |  [MSFT_MACD] / [0.90]              | ... 
            | B_OPEN     |  [AAPL_CLOSE] / [0.99]             | ...
        
            borda_count_on_all_tickers(top_df, top_X = 2) ==> 
            | Ticker  |    Top_1    |    Top_2   ...
            |    A    | (MSFT, 20)  |  (TMO, 10) ...
            |    B    | (AAPL, 18)  | ...
    ''' 
    def borda_count_on_ticker(self, top_df, top_X : int, ticker : str, all_tickers : list):
         # (0) Rename indicator column to ticker and only leave the ticker (no need for the feature name after _)
        top_df = top_df.rename(columns = {'Indicator' : 'ticker'}, inplace = False)
        top_df['ticker'] = top_df['ticker'].str.split('_').str[0]

         # (1) Initialize dictionary where key = ticker, value = votes it's key recieved 
        votes : dict = dict()
        for ticker in all_tickers:
            votes[ticker] = 0

        # (2) Get df w/correlations of only our ticker            
        corr_of_ticker = getSymbolData(df = top_df, symbol = ticker) 

        # (3) Conduct borda count by iterating through 'corr_of_ticker'
        (rows, columns) = corr_of_ticker.shape
        for i in range (0, rows): 
            curr_row     = corr_of_ticker.iloc[i]
            curr_row_top = [] # len(curr_row_top) == top_X

            for j in range (1, top_X + 1):
                curr_col_name = f'Top_{j}'                  # (i)   Rebuild column name
                string_at_col = curr_row[curr_col_name]     # (ii)  Get column entry (ex: "[B_OPEN] / [0.9]")
                ticker_at_cel = string_at_col.split("_")[0] # (iii) Extract the ticker only -> '[B'
                ticker_at_cel = ticker_at_cel.strip('[')    # (iv)  Remove '[' -> 'B'
                curr_row_top.append(ticker_at_cel)          # (v)   Insert ticker to row's top -> [B, ...]

            # (4) Give points in decreasing order to all the tickers in 'curr_row_top'
            points = top_X
            for ticker in curr_row_top:
                votes[ticker] += points 
                points -= 1
        
        # (5) Find 'top_X' tickers with the most votes in vote dictionary and put them in data frame
        sorted_dict_by_votes = dict(sorted(votes.items(), key=lambda item: item[1], reverse = True))
        top_tickers_for_ticker = pd.DataFrame()
        top_tickers_for_ticker['Ticker'] = ticker
        i = 1
        for ticker in list(sorted_dict_by_votes.keys())[: top_X]:
            top_tickers_for_ticker[f'Top_{i}'] = (ticker, sorted_dict_by_votes[ticker])
            i += 1
        
        return top_tickers_for_ticker
    
    def borda_count_on_all_tickers(self, top_df, top_X : int, all_tickers : list): 
        borda_count_results = pd.DataFrame()
        for ticker in all_tickers:
            top_X_for_ticker = self.borda_count_on_ticker(top_df = top_df, top_X = top_X, ticker = ticker, all_tickers = all_tickers)
            borda_count_results = top_X_for_ticker if borda_count_results.empty else pd.concat([borda_count_results, top_X_for_ticker], ignore_index = True )
        return borda_count_results
    
    '''
        feature_by_feature_on_ticker - function takes in ticker, corr_matrix and number of 
        pairs to identify and conducts the following voting rule:

        1.  Identify Rows: Finds all rows in the correlation matrix that start with the given ticker.
                        | A_open | A_close  | ... | B_open | B_close | ... 
                A_open  |  1.0   |    0.9   |     |  0.3   |  0.2  
                A_close |  0.9   |    1.0   |     |  0.35  |  0.5
                
                => size_of_matrix = features_of_ticker x all_features
        
        2.  Creates matrix with rows being all the features identified in step 1 and columns being 
            all tickers equivalent feature:

                ticker's features| A_feature | B_feature | ... |
                A_open           |  1.0      |    0.4    | ... |     
                A_close          |  1.0      |    0.7    | ... |   

                => size_of_matrix = features of ticker x number of tickers 
        
        3.  Get each ticker's feature_by_feature score by getting the mean of each column 
 
                ticker's features|   A  |  B  | ... |
                A_open           |  1.0 | 0.4 | ... |   => scores : dict = {A: 1.0, B: 0.55, ...}     
                A_close          |  1.0 | 0.7 | ... |       
        
        4. Sort dictionary and return the top pairs requested in data frame like such:
            * we exclude the ticker it self by setting its score in dict to -1 
                Ticker | Top_1 | Top_2 | ... | 
                  A        B      ...
        
        feature_by_feature_on_all_tickers - runs feature_by_feature analysis on all tickers to 
        return dataframe like such:
                Ticker | Top_1      | Top_2 | ... | 
                A      | (B,0.9)    | CAT ...
                AAPL   | (MSFT,0.8) | NVIDIA ...
                ⋮           ⋮         ⋮      
    '''
    def feature_by_feature_on_ticker(self, corr_matrix, ticker : str, top : int, tickers : list):
        # (1) Identify sub-matrix of correlation_matrix of features that have ticker as prefix 
        ticker_matrix = corr_matrix.filter(regex = f'^{ticker}_')
        ticker_matrix = ticker_matrix.T # transpose

        # (2) Create equivalent feature matrix 
        cols = ['ticker feature'] + tickers
        eq_matrix = pd.DataFrame(columns= cols)

        for index, row in ticker_matrix.iterrows():
            # (i)  Get indicator out of feature (AAPL_ROC_14 -> close)  
            indicator = index.split('_')[1:]
            indicator = "_".join(indicator)

            # (ii) Get columns of ticker_matrix w/identified indicator
            indicator_cols = [col for col in ticker_matrix.columns if isinstance(col, str) and col.endswith(f'_{indicator}')]  
            new_row_for_eq_matrix : dict = {}
            new_row_for_eq_matrix['ticker feature'] = [index]

            for col in indicator_cols: 
                curr_ticker = col.split('_')[0]                 # (iii) Get the ticker of each of the indicator columns 
                corr_val = row[col]                             # (iv)  Get the correlational value of that indicator 
                new_row_for_eq_matrix[curr_ticker] = [corr_val] # (v)   Create new entry in dict so we get {ticker feature: A_close, A: 1.0, B: 0.4, ...}
            
            eq_matrix = pd.concat([eq_matrix, pd.DataFrame(new_row_for_eq_matrix)], ignore_index = True)
            break  
        
        # (3) Get each ticker's feature by feature score by taking mean of their column in eq_matrix
        scores : dict = {}
        for column in eq_matrix:
            if column != 'ticker feature':
                scores[column] = round(eq_matrix[column].mean() * 100, 5)
        scores[ticker] = -1


        # (4) Sort dictionary by score and return the top pairs requested in data frame 
        sorted_scores : list = sorted(scores.items(), key = lambda item: item[1], reverse = True)
        top_scores : list = sorted_scores[:top]
        ticker_col = pd.DataFrame([ticker], columns = ['ticker'])
        other_cols = pd.DataFrame([top_scores], columns=[f'Top_{i+1}' for i in range(top)])
        final = pd.concat([ticker_col, other_cols], axis = 1)

        return final 
    
    def feature_by_feature_on_all_tickers(self, corr_matrix, top : int, tickers : list):
        results = pd.DataFrame()
        for ticker in tickers:
            top_of_ticker = self.feature_by_feature_on_ticker(corr_matrix, ticker, top, tickers)
            results = pd.concat([results, top_of_ticker], ignore_index = True) if results.empty == False else top_of_ticker        
        return results
    
