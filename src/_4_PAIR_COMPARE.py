from _0_helper_functions import *
'''
                                                << PAIR_COMPARE (module) >>
        => Class Initialization Inputs: None

    ------------ User-facing Methods --------------
        (1) compare_pairs(file_1_path: str, file_1_num_features: int, file_2_path: str, file_2_num_features: int):
            => Compares pairs identified in two different studies.
            => Inputs:
                file_1_path         : str  | Path to the first data frame.
                file_1_num_features : int  | Number of features used in the first study.
                file_2_path         : str  | Path to the second data frame.
                file_2_num_features : int  | Number of features used in the second study.
            => Outputs:
                DataFrame | DataFrame listing the intersection, union, and exclusive pairs from both files.
                | Intersection        |     Union       | Exclusive Pairs - File 1 | Exclusive Pairs - File 2
                (A-B, F1: 10, F2: 20)  (AAPL-MSFT, 30) ...

    ------------ Backend Methods --------------
        (1) read_pair_files(file_1_path: str, file_2_path: str):
            => Reads data from pair files and verifies their dimensions.
            => Inputs:
                file_1_path : str | Path to the first data frame.
                file_2_path : str | Path to the second data frame.
            => Outputs:
                tuple | A tuple containing the two data frames (f1, f2).

        (2) get_votes_per_ticker(num_of_features_per_ticker: int, num_of_top_pairs_per_ticker: int):
            => Calculates the number of votes each ticker awarded based on the number of features.
            => Inputs:
                num_of_features_per_ticker  : int | Number of features per ticker.
                num_of_top_pairs_per_ticker : int | Number of top pairs per ticker.
            => Outputs:
                int | Total votes per ticker.

        (3) transform_votes_to_percentage(pair_file, votes_per_ticker: int):
            => Converts raw votes of pairs into percentage values.
            => Inputs:
                pair_file       : DataFrame | Data frame with pair votes.
                votes_per_ticker: int       | Total votes per ticker.
            => Outputs:
                DataFrame | Data frame with votes transformed into percentages.

        (4) get_all_pairs_in_df(df):
            => Returns a set of all pairs in the data frame.
            => Inputs:
                df : DataFrame | Data frame with pairs data.
            => Outputs:
                set | Set of pairs.

        (5) pairs_dict_from_set(pairs_set: set):
            => Converts a set of pairs into a dictionary with percentages.
            => Inputs:
                pairs_set: set | Set of pairs.
            => Outputs:
                dict | Dictionary with pairs as keys and percentages as values.

        (6) find_intersection(pairs_dict_1: dict, pairs_dict_2: dict):
            => Finds pairs common to both dictionaries.
            => Inputs:
                pairs_dict_1: dict | First pairs dictionary.
                pairs_dict_2: dict | Second pairs dictionary.
            => Outputs:
                list | List of common pairs in descending order of their highest percentage.

        (7) find_difference(pairs_in_dict: dict, pairs_not_in_dict: dict):
            => Finds pairs in one dictionary not present in the other.
            => Inputs:
                pairs_in_dict    : dict | Dictionary with pairs.
                pairs_not_in_dict: dict | Dictionary without pairs.
            => Outputs:
                list | List of exclusive pairs in ascending order of their percentage.

        (8) find_union(pairs_dict_1: dict, pairs_dict_2: dict):
            => Finds all pairs present in either dictionary.
            => Inputs:
                pairs_dict_1: dict | First pairs dictionary.
                pairs_dict_2: dict | Second pairs dictionary.
            => Outputs:
                list | List of all pairs with their percentages in descending order.
'''

class PAIR_COMPARE(): 
    '''
        compare_pairs - This function takes two file paths to data frames that contain pairs 
        identified in different studies. It also requires the number of features used in those 
        studies to find the pairs.
        
            => Output: a data frame in the following format:
            | Intersection        |     Union       |Exclusive Pairs - File 1 | Exclusive Pairs - File 2
            (A-B, F1: 10, F2: 20)  (AAPL-MSFT, 30) ...
            
            * cell format: ('pair' : str, correlational strength percent: float) 
            * correlational strength percent = percentage of votes pair had out of total votes availale in borda count.

            => REQUIRES: The files must contain the same number of top pairs (e.g., both files should 
            have the top 10 pairs or the top 3 pairs).
    '''
    def compare_pairs(self, file_1_path : str, file_1_num_features : int, voting_rule_1 : str, 
                            file_2_path : str, file_2_num_features : int, voting_rule_2 : str): 
        # (1) Read data
        f1, f2 = self.read_pair_files(file_1_path = file_1_path, file_2_path = file_2_path)

        # (2) Case on Voting Method 
        pair_df_1 = self.scale_borda_count(f1, file_1_num_features) if voting_rule_1 == 'Borda Count' else f1.copy()
        pair_df_2 = self.scale_borda_count(f2, file_2_num_features) if voting_rule_2 == 'Borda Count' else f2.copy()
        
        if voting_rule_1 == 'Feature by Feature':
            pair_df_1.set_index('ticker', inplace = True)
        
        if voting_rule_2 == 'Feature by Feature':
            pair_df_2.set_index('ticker', inplace = True)
       
        # (3) Create a set for each pair df that contains all pairs (and their strengths) in that df 
        pairs_in_df_1 : set = self.get_all_pairs_in_df(df = pair_df_1) # ex: {(AAPL-MSFT, 20), (MSFT-AMD, 10)...}
        pairs_in_df_2 : set = self.get_all_pairs_in_df(df = pair_df_2) 

        # (4) Transform pairs sets into pairs dict where keys = pairs and values = percentage 
        pairs_in_df_1 : dict = self.pairs_dict_from_set(pairs_set = pairs_in_df_1)
        pairs_in_df_2 : dict = self.pairs_dict_from_set(pairs_set = pairs_in_df_2)

        # (5) Find intersection, union and exclusive pairs per file 
        results : dict = {
                            'Intersection'           : self.find_intersection(pairs_dict_1 = pairs_in_df_1, pairs_dict_2 = pairs_in_df_2),
                            'File 1 Exclusive Pairs' : self.find_difference(pairs_in_dict = pairs_in_df_1, pairs_not_in_dict = pairs_in_df_2),
                            'File 2 Exclusive Pairs' : self.find_difference(pairs_in_dict = pairs_in_df_2, pairs_not_in_dict = pairs_in_df_1),
                            'Union'                  : self.find_union(pairs_dict_1 = pairs_in_df_1 , pairs_dict_2 = pairs_in_df_2)
                         }
        
        # (7) Find biggest column, and populate others to get same size arrays (so we can create pandas df)
        max_length = 0
        for column_title, value_list in results.items():
            if len(value_list) > max_length:
                max_length = len(value_list)

        for column_title, value_list in results.items():
            size_difference = max_length - len(value_list)
            scaled_up_list  = value_list + ([np.nan] * size_difference)
            results[column_title] = scaled_up_list
        
        return pd.DataFrame(results) 

    ''' 
        read_pair_files - reads data of pair files and fails if files have different dimensions 
        (for instance if one file has top 10 pairs while the  other has top 3)
    '''
    def read_pair_files(self, file_1_path : str, file_2_path : str):
        f1 = pd.read_csv(file_1_path)
        f2 = pd.read_csv(file_2_path)

        if f1.shape[1] != f2.shape[1]:
            print('ERROR: Pair Files do not Have the Same Dimensions!')
            raise AssertionError

        return f1, f2
    
    '''
        scale_borda_count - converts borda counts votes into votes/number_of_votes_each_ticker_awarded
    '''
    def scale_borda_count(self, file, file_num_features):
        # (1) Get number of top pairs each ticker has (X)
        X = 0
        for column in file.columns:
            col_list = column.split('_')
            if col_list[0] == 'Top':
                X += 1

        # (2) Transform votes each pair recieved to percentage of total votes each ticker awarded (if borda count)
        votes_per_ticker = self.get_votes_per_ticker(num_of_features_per_ticker = file_num_features, num_of_top_pairs_per_ticker = X)
        scaled_df = self.transform_votes_to_percentage(pair_file = file, votes_per_ticker = votes_per_ticker)

        return scaled_df

    ''' 
        get_votes_per_ticker - calculates number of votes each ticker awarded based on the 
        number of features of the file. Initially each ticker had a certain number of features
        each with it's top correlated features from other tickers. 
            total votes = num(features of ticker) * summation(1, n) where n = the top pairs per ticker 
    '''
    def get_votes_per_ticker(self, num_of_features_per_ticker : int, num_of_top_pairs_per_ticker):
        summation : int = 0
        for i in range (1, num_of_top_pairs_per_ticker + 1):
            summation +=i

        return num_of_features_per_ticker * summation 

    '''
        transform_votes_to_percentage - turns raw votes of pairs into percentage in regards to 
        the ticker's total awarded votes. 
            => votes -> (vote_of_pair / votes_per_ticker) * 100
    '''
    def transform_votes_to_percentage(self, pair_file, votes_per_ticker : int): 
        # (1) Get a copy of passed in df but without the ticker column 
        if 'Unnamed: 0' in pair_file:
            pair_file.drop('Unnamed: 0', axis = 1, inplace = True)
        pair_file_no_tic_col = pair_file.drop('ticker', axis = 1, inplace = False)

        # (2) Define mapping helper function to map all cells in df 
        def votes_to_per_aux(cell_input, votes_per_ticker : int):
            cell_input = str(cell_input)                               # (i)   Turn cell into string      -> '(AAPL, 20)'
            cell_input = cell_input.replace('(', "").replace(')', "")  # (ii)  Remove brackets            -> 'AAPL, 20'
            tv = cell_input.split(',')                                 # (iii) Create ticker - votes list -> ['AAPL', '20']
            tv[1] = round((float(tv[1]) / votes_per_ticker) * 100, 1)  # (iv)  Turn votes to percentage   -> ['AAPL', 3.1]
            mapped_string = f'({tv[0]}, {tv[1]})'                      # (v)   Rebuild the mapped string  -> '(AAPL, 3.1)'
            return mapped_string
        
        # (3) Map columns using helper function 
        pair_file_no_tic_col = pair_file_no_tic_col.applymap(lambda x: votes_to_per_aux(cell_input = x, votes_per_ticker = votes_per_ticker))
        
        # (4) Re-insert ticker column, set is as index and return 
        final = pair_file_no_tic_col.copy()
        final['ticker'] = pair_file['ticker'].tolist()
        final.set_index('ticker', inplace = True)
        return final 

    '''
        get_all_pairs_in_df - returns a set with all the pairs in the data frame. 
        {('AAPL-MSFT', 30), ('AMD-A', 20) ...}

        REQUIRES: 'ticker' column to be index of df 

        pairs_dict_from_set - converts pairs set to a pairs dictionary where keys 
        are pairs and values are their percentage
    '''
    def get_all_pairs_in_df(self, df):
        output_set : set = set()
        # Iterate through all rows of df 
        for ticker in df.index: 
            list_of_corr = df.loc[ticker].tolist()   # (i) Get list of correlated tickers to ticker -> [(AMD, 30), (ABT, 27)...]

            for item in list_of_corr: 
                # (ii) Format correlated tickers '(AMD, 30)' -> 'AMD,30'
                item = item.replace('(', '').replace(')', '').replace("'", "").replace(" ", "")

                # (iii) Split each item & form tupple like such: ('AAA-BBB', 20)
                item = item.split(',') # [AMD, 30]
                
                new_pair = (f'{ticker}-{item[0]}', float(item[1]))

                # (iv) add pair to output set
                output_set.add(new_pair)
        
        return output_set

    def pairs_dict_from_set(self, pairs_set : set): 
        pairs_dict : dict = dict()
        for pair_tupple in pairs_set:
            pair    : str   = pair_tupple[0]                # (i)   Get pair from tupple
            percent : float = pair_tupple[1]                # (ii)  Get percetange val from tupple 
            symbols : list  = pair.split('-')               # (iii) Get symbols by splitting pair at '-'
            reverse : str   = f'{symbols[1]}-{symbols[0]}'  # (iv)  Build the reverse of the pair (A-B -> B-A)
            key = pair if pair < reverse else reverse       # (v)   Get the key we will use to store pair by ordering pair in alphabetical order

            # (vi) If key is already in dict, keep greatest percentage as the value 
            if key in pairs_dict: 
                biggest = max(pairs_dict[key], percent)
                pairs_dict[key] = biggest
            else:
                pairs_dict[key] = percent
        
        return pairs_dict
    
    '''
        find_intersection - finds the pairs two dictionaries have in common. Returns 
        a list of these pairs (in ascending order based on their highest percentage value).
    '''
    def find_intersection(self, pairs_dict_1 : dict, pairs_dict_2 : dict): 
        # (1) Get keys (pairs) that are in both dictionaries 
        pairs_1 : set = set(pairs_dict_1.keys())
        pairs_2 : set = set(pairs_dict_2.keys())
        intersection : set = pairs_1 & pairs_2

        # (2) For each pair in intersection create a tripple like such ('AAA-BBB', percentage 1, percentage 2)
        final_intersection : list = []
        for pair in intersection:
            tripple = (pair, pairs_dict_1[pair], pairs_dict_2[pair])
            final_intersection.append(tripple)
        
        # (3) Sort intersection by the max of the two percentages 
        final_intersection.sort(key = lambda x: max(x[1:]), reverse = True)

        return final_intersection
    
    '''
        find_difference - finds the pairs that in 'pairs_in_dict' that are not in 'pairs_not_in_dict'.
        Returns a list of these pairs (in ascending order based on each pair's percentage) 
    '''
    def find_difference(self, pairs_in_dict : dict , pairs_not_in_dict : dict):
        # (1) Find exclusive set of pairs
        pairs_1 : set = set(pairs_in_dict.keys())
        pairs_2 : set = set(pairs_not_in_dict.keys())
        exclusive_pairs : set = pairs_1 - pairs_2

        # (2) Build list of tupples ('AAA-'BBB', percentage)
        exclusive : list = []
        for pair in exclusive_pairs:
            pair_and_percent = (pair, pairs_in_dict[pair])
            exclusive.append(pair_and_percent)
        
        # (3) Sort it in ascending order and return 
        exclusive.sort(key = lambda x: x[1], reverse = True)
        return exclusive
    
    '''
        find_union - gets all the pairs in both data frames.  
    '''
    def find_union(self, pairs_dict_1 : dict , pairs_dict_2 : dict):
        pairs_1 : set = set(pairs_dict_1.keys())
        pairs_2 : set = set(pairs_dict_2.keys())
        union   : set = pairs_1 | pairs_2

        # (1) Iterate through union and augment each pair with their percentages 
        final_union : list = []
        for pair in union:
            if (pair in pairs_1) and (pair in pairs_2):
                final_union.append((pair, [pairs_dict_1[pair], pairs_dict_2[pair]]))

            elif pair in pairs_1:
                final_union.append((pair, [pairs_dict_1[pair]]))
            
            else:
                final_union.append((pair, [pairs_dict_2[pair]]))
        
        # (2) Turn final union set into sorted list 
        final_union.sort(key = lambda x: max(x[1]), reverse = True)

        # (3) Flatten all lists to transform (pair, [corrs...]) =>  (pair, corr) or (pair, corr1, corr2)
        final : list = []
        for tupple in final_union:
            pair = tupple[0]
            list_of_percentages = tupple[1]
            if len(list_of_percentages) == 2:
                new = (pair, list_of_percentages[0], list_of_percentages[1])
            else:
                new = (pair, list_of_percentages[0])
            final.append(new)
            
        return final
