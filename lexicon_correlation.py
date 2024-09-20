# Set up the dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
                                         << LexiconCorrelation (module) >>
    => Class Initialization Inputs:
        (1) path_to_file        : str  | Path to the CSV file containing the data.
        (2) path_to_save        : str  | Path to save the computed correlation matrix.
        (3) display_heatmap     : bool | Flag to indicate whether to display the heatmap (default: True).

    ------------ User-facing Methods --------------
        (1) run_correlation():
            => Computes the correlation matrix for sentiment scores, optionally displays it, and saves it to a specified file.
            => Inputs: None
            => Outputs: None

"""

class LexiconCorrelation():
    def __init__(self, path_to_file: str, 
                 path_to_save: str,
                 display_heatmap: bool = True):
        # (1) Initialize parameters into class
        self.path_to_file = path_to_file        # Path to the CSV file containing the data
        self.path_to_save = path_to_save        # Path to save the computed correlation matrix
        self.data = pd.read_csv(path_to_file)   # Load the data from the CSV file
        self.display_heatmap = display_heatmap  # Flag to indicate whether to display the heatmap

    '''
        display_heatmap - optionally visualizes the correlation matrix as a heatmap.
        => Inputs:
            correlation_matrix : DataFrame | Correlation matrix to display.
        => Outputs: None
    '''
    def display(self, correlation_matrix):
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Sentiment Scores')
        plt.show()

    '''
        run_correlation - computes the correlation matrix for sentiment scores, optionally displays it, and saves it to a specified file.
        => Inputs: None
        => Outputs: None
    '''
    def calculate_returns(self, returns):
        # Load stock data
        stock_df = pd.read_csv('filtered_stock_data.csv')    # for future optimizations can be just concatenated for the 

        for day in range(1, returns + 1):
            # Group by ticker and calculate returns within each group
            stock_df['tomorrow_close'] = stock_df.groupby('tic')['close'].shift(-day)
            stock_df[f'return_D-{day}'] = (stock_df['tomorrow_close'] - stock_df['close']) / stock_df['close'] * 100

            # Remove the auxiliary column if no longer needed
            stock_df.drop(columns=['tomorrow_close'], inplace=True)
        
        # Ensure the date columns are explicitly converted to datetime format
        stock_df['datadate'] = pd.to_datetime(stock_df['datadate'])
        stock_df.rename(columns={'datadate': 'date'}, inplace=True)
        
        return stock_df 

    def run_correlation(self, include_return = None):
        if include_return is not None: 
            stock_df = self.calculate_returns(include_return)
            self.data['date'] = pd.to_datetime(self.data['date'])
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            merged_df = pd.merge(self.data, stock_df, on=['date', 'tic'], how='left')
            merged_df = merged_df.drop(columns=["conm","volume","close","high","low","open"])
        else: 
            merged_df = self.data
        # Compute the correlation matrix for sentiment scores, dropping non-numeric columns
        correlation_matrix = merged_df.drop(columns=['tic', 'date']).corr()
        
        # Display the heatmap if the flag is set
        if self.display_heatmap: 
            self.display(correlation_matrix)
        
        # Save the correlation matrix to a CSV file
        correlation_matrix.to_csv(self.path_to_save)

    