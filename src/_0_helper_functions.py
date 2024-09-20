''' 
    Downloads required: 
        1.) pip install pandas
        2.) pip install ta 
        3.) pip install statsmodels
        4.) pip install scikit-learn
        5.) pip install yfinance
'''
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


from datetime import datetime 
from itertools import product

import sklearn_preprocessing
from sklearn.preprocessing import MinMaxScaler

import math

import ta

import yfinance as yf

import multiprocessing as mp
from multiprocessing import Pool, cpu_count


# ================================================================================================ #
'''
    Helper Functions:
        (0) getTickers(wrds_data):
                => returns list of tickers of data 

        (1) getSymbolData(df, symbol)
                => returns dataframe with data of only one specific symbol
        
        (2) dropTickers(df, tickersToKeep) 
                => returns a data frame with data of only the specified tickers

        (3) get_NaN_of_symbol(df, symbol)
                => returns the percentage of nan in relevant columns of a symbol
        
        (4) emptyTickers(df, symbols, threshold)
                => returns a list of all tickers whose nan% is above threshold
        
        (5) rows_per_ticker(df, symbols)
                => returns list w/data of rows per ticker

        (6) periodCorrector(dataframe, period, indicatorName, listOfTickers)
                => returns a data frame setting the first "period" items to 
                   NaN in "indicatorName" Column

        (7) mergeDataFrames(df_1, df_2, column_to_merge)
                => returns df that merges rows of two DataFrames based on a 
                   specified column and returns the resulting DataFrame
                => ex: DataDate | A_indicator 1 | A_indicator 2 | B_indicator 1 | B_indicator 2
        
        (8) getClosingPrices(df, symbol) 
                => returns the closing prices of a symbol from data frame 

        (9) getSymbolCorrelation(symbol_1, symbol_2, price_data)
                => returns the correlation between symbols using their closing price info
        
        (10) crossover(prev_zscore, curr_zscore, zscore_tp):
                => returns true if there has been a cross over in zscore 
        
        (11) get_data_in_date_range(df, start_date, end_date):
                => returns df but with rows of data only between specified interval 
        
        (12) get_tickers_from_corr_matrix(corr_matrix):
                => returns a list of all tickers in correlation matrix 
 
'''

''' =========================================== (0) =========================================== '''
def getTickers(wrds_data):
  allTickers_with_dup = wrds_data['ticker']
  allTickers_no_dup = sorted(list(set(allTickers_with_dup)))
  return allTickers_no_dup

# 'getTickers' RETURNS A LIST OF UNIQUE TICKERS
def getTickers2(df, nameOfColumnWithTicker):
  allTickers_with_dup = df[nameOfColumnWithTicker]
  allTickers_no_dup = sorted(list(set(allTickers_with_dup)))
  return allTickers_no_dup

''' =========================================== (1) =========================================== '''
def getSymbolData(df, symbol):
    # Get Rows of the Ticker
    ticker_rows = df[df['ticker'] == symbol]
    return ticker_rows

def getSymbolData2(df, symbol, nameOfColumnWithTicker):
    # Get Rows of the Ticker
    ticker_rows = df[df[nameOfColumnWithTicker] == symbol]
    return ticker_rows

''' =========================================== (2) =========================================== '''
def dropTickers(df, tickersToKeep):
    # Create new row w/in top 300 flag
    new_df = df.copy()
    new_df['KeepFlag'] = new_df['ticker'].isin(tickersToKeep)

    # Drop all Rows whose Keep is false
    new_df = new_df[new_df['KeepFlag']]  # Keep only rows where Keep Flag is True

    # Drop Keep Flag Column
    new_df.drop('KeepFlag', axis = 1, inplace = True )

    return new_df

''' =========================================== (3) =========================================== '''
def get_NaN_of_symbol(df, symbol):
    relevant_columns = ['open', 'high', 'low','close', 'volume']
    df_of_ticker = getSymbolData(df, symbol)
    nan_of_symbol = df_of_ticker[relevant_columns].isna().mean() * 100
    return nan_of_symbol

''' =========================================== (4) =========================================== '''
def emptyTickers(df, symbols, threshold = 0.9):
    tickers_to_remove = []

    for ticker in symbols:
        # Get Data Frame of All Tickers
        df_of_ticker = getSymbolData(df, ticker)

        # Check the NaN percetage per ticker
        relevant_columns = ['open', 'high', 'low','close', 'volume']
        na_percent_of_ticker = df_of_ticker[relevant_columns].isna().mean()

        # If percentage of NaN is over threshold, add ticker to remove list
        if any(na_percent_of_ticker > threshold): # na is an obj w/ na% of open, high, etc...
            tickers_to_remove.append(ticker)

    return tickers_to_remove

''' =========================================== (5) ==========================================='''
def rows_per_ticker(df, symbols):
    # Intialize a dictionary where keys = #rows and value = #tickers w/ that amount of rows
    rows_dict = {}

    for ticker in symbols:
        # Get how many rows that ticker has:
        (rows, _) = getSymbolData(df, ticker).shape

        if rows in rows_dict:
            rows_dict[rows] = rows_dict[rows] + 1
        else:
            rows_dict[rows] = 1

    # Get amount of rows with most associated tickers
    max_key = -1
    max_val = -1
    for num_rows in rows_dict:
      ticker_with_that_many_rows = rows_dict[num_rows]

      if ticker_with_that_many_rows > max_val:
        max_val = ticker_with_that_many_rows
        max_key = num_rows

    # Get Percentange of Tickers with number of rows = most common number of rows
    total_tickers = len(symbols)
    if total_tickers == 0:
        percentage = 0
    else:
        percentage = max_val / total_tickers * 100

    # Build List to Return
    results = [None, None, None, None]
    results[0] = rows_dict  # dictionary with all results
    results[1] = max_key    # Most Common Number of Rows
    results[2] = max_val    # Number of Tickers that have the Most Common Number of Rows
    results[3] = round(percentage, 1) # Percent of Tickers that have the Most Common Number of Rows

    return results
''' =========================================== (6) ==========================================='''
def periodCorrector(dataframe, period, indicatorName, listOfTickers):
    for ticker in listOfTickers:
        ticker_rows = dataframe[dataframe['ticker'] == ticker]
        idxs_to_fix = ticker_rows.index[:period-1]
        dataframe.loc[idxs_to_fix, indicatorName] = np.nan
    return dataframe

''' =========================================== (7) ==========================================='''
def mergeDataFrames(df_1, df_2, column_to_merge):
    if df_1.empty: 
        return df_2
    elif df_2.empty:
        return df_1
    else:
        # Merge df_1 and df_2
        merged_df = pd.merge(df_1, df_2, on = column_to_merge)
        return merged_df

''' =========================================== (8) ==========================================='''
def getClosingPrices(df, symbol):
    # Get symbol data
    symbol_data = getSymbolData(df, symbol)

    # Get closing prices
    closing_prices = pd.Series(symbol_data['close'])
    closing_prices.reset_index(drop = True, inplace = True)

    return closing_prices

''' =========================================== (9) ==========================================='''
def getSymbolCorrelation(symbol_1, symbol_2, price_data):
    # Get Closing Prices
    close_prices_1 = getClosingPrices(price_data, symbol_1)
    close_prices_2 = getClosingPrices(price_data, symbol_2)

    # Calculate the correlation
    correlation = close_prices_1.corr(close_prices_2)

    return correlation

''' =========================================== (10) ==========================================='''
def crossover(prev_zscore, curr_zscore, zscore_tp):
    if prev_zscore < zscore_tp and curr_zscore >= zscore_tp:
        return True

    elif prev_zscore > zscore_tp and curr_zscore <= zscore_tp:
        return True

    elif curr_zscore == zscore_tp:
        return True

    else:
        return False
    
''' =========================================== (11) =========================================== '''
def get_data_in_date_range(df, start_date : str, end_date : str):
    df['date'] = pd.to_datetime(df['date']) #(i) Convert date to datetime format 
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df = filtered_df.reset_index(drop = True)
    return filtered_df

''' =========================================== (12) =========================================== '''
def get_tickers_from_corr_matrix(corr_matrix):
    columns = corr_matrix.columns 
    tickers : set = set()
    for column in columns: 
        ticker = column.split('_')[0]
        tickers.add(ticker)
    return list(tickers)


