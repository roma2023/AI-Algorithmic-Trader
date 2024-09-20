# Project Modules

## 1. STOCK_PRICES Module

The `STOCK_PRICES` module helps retrieve and clean stock price data.

### Key Features

- **Retrieve Stock Data**: Fetches stock prices from Yahoo Finance.
- **Clean Data**: Cleans the stock price data to ensure accuracy.

### How to Use

1. **Fetch Data**:
    ```python
    stock_prices.get_data_from_yf(tickers, start_date, end_date)
    ```
    - **tickers**: List of stock symbols.
    - **start_date**: Start date (e.g., '2023-01-01').
    - **end_date**: End date (e.g., '2024-01-01').

2. **Clean Data**:
    ```python
    stock_prices.get_clean_stock_data(tickers_list, path_to_prices_df, yf_start, yf_end)
    ```
    - Can use a list of tickers or a path to a CSV file.

## 2. INDICATORS Module

The `INDICATORS` module adds technical indicators to the stock data.

### Key Features

- **Technical Indicators**: Adds various indicators like RSI, MACD, Bollinger Bands, etc.

### How to Use

1. **Initialize Data**:
    ```python
    indicators.initializeMainDataFrame()
    ```

2. **Add RSI**:
    ```python
    indicators.addRSI(period)
    ```
    - **period**: Number of days for the calculation.

## 3. PAIR_FINDER Module

The `PAIR_FINDER` module identifies and ranks correlated stock pairs.

### Key Features

- **Find Pairs**: Identifies top correlated stock pairs.
- **Voting Rules**: Uses methods like Borda Count to rank pairs.

### How to Use

1. **Create Correlation Matrix**:
    ```python
    pair_finder.create_corr_matrix(path_to_prices_or_sent, path_to_save)
    ```

2. **Get Top Pairs**:
    ```python
    pair_finder.get_top_pairs_of_all_tickers(corr_matrix)
    ```

## 4. PAIR_COMPARE Module

The `PAIR_COMPARE` module compares pairs identified in different studies.

### Key Features

- **Compare Pairs**: Finds common and unique pairs between studies.

### How to Use

1. **Compare Pairs**:
    ```python
    pair_compare.compare_pairs(file_1_path, file_1_num_features, file_2_path, file_2_num_features)
    ```

## 5. DATA_MANIPULATOR Module

The `DATA_MANIPULATOR` module manages and manipulates stock price data.

### Key Features

- **Manage Data**: Loads and processes stock data.
- **Calculate Metrics**: Calculates hedge ratios, spreads, and cointegration.

### How to Use

1. **Initialize Data**:
    ```python
    data_manipulator.initialize_data()
    ```

2. **Get Data Between Dates**:
    ```python
    data_manipulator.get_data_between_dates(symbol, start_date, end_date)
    ```

## 6. ZSCORE_STRATEGY Module

The `ZSCORE_STRATEGY` module helps implement a trading strategy based on Z-scores.

### Key Features

- **Trading Strategy**: Uses Z-scores to identify trading signals.
- **Backtesting**: Tests the strategy on historical data.

### How to Use

1. **Initialize Strategy**:
    ```python
    strategy = ZSCORE_STRATEGY(symbol_1, symbol_2, path_to_prices_df, start_date, end_date, zscore_period, zscore_entry, zscore_stop_loss, zscore_take_profit, use_coint_check, coint_check_length, use_days_stop, days_for_stop)
    ```

2. **Build Backtester Data**:
    ```python
    strategy.build_backtester_df()
    ```

## 7. PAIRS_BACKTESTER Module

The `PAIRS_BACKTESTER` module helps backtest the Z-score trading strategy.

### Key Features

- **Backtesting**: Simulates the strategy on historical data.
- **Performance Metrics**: Provides detailed backtest results.

### How to Use

1. **Initialize Backtester**:
    ```python
    backtester = PAIRS_BACKTESTER(strategy_obj, initial_capital, fixed_allocation, risk_per_trade)
    ```

2. **Run Backtest**:
    ```python
    results = backtester.backtest(show_output=True)
    ```

## 8. OPTIMIZER Module

The `OPTIMIZER` module optimizes trading strategy parameters for the best performance.

### Key Features

- **Parameter Optimization**: Finds the best parameters for the strategy.
- **Result Analysis**: Analyzes and displays optimization results.

### How to Use

1. **Initialize Optimizer**:
    ```python
    optimizer = OPTIMIZER(pairs_list, start_date, end_date, path_to_prices_df, strategy_name, parameters_ranges, initial_capital_per_backtest, fixed_allocation_per_backtest, risk_per_trade)
    ```

2. **Run Optimization**:
    ```python
    top_results = optimizer.optimize(results_to_optimize, results_to_show=10, show_results=True)
    ```

## 9. PAIR_RANKER Module

The `PAIR_RANKER` module ranks pairs based on their statistical metrics.

### Key Features

- **Pair Ranking**: Ranks pairs using specified weights for various metrics.
- **Statistical Analysis**: Provides detailed statistics for each pair.

### How to Use

1. **Rank Pairs**:
    ```python
    ranked_pairs = pair_ranker.rank_pairs(path_to_pairs_csv, start_date, end_date, weights_per_col)
    ```

## 10. LexiconCorrelation Module

The `LexiconCorrelation` module computes correlations between sentiment scores.

### Key Features

- **Correlation Analysis**: Calculates and visualizes correlations between sentiment scores.

### How to Use

1. **Run Correlation**:
    ```python
    lexicon_corr = LexiconCorrelation(path_to_file, path_to_save, display_heatmap=True)
    lexicon_corr.run_correlation()
    ```

## 11. SentimentAnalysis Module

The `SentimentAnalysis` module performs sentiment analysis using various models and lexicons.

### Key Features

- **Sentiment Analysis**: Analyzes sentiment using multiple models like FinBERT, VADER, and more.
- **Daily Sentiment Scores**: Aggregates sentiment scores on a daily basis.

### How to Use

1. **Initialize Sentiment Analysis**:
    ```python
    sentiment_analysis = SentimentAnalysis(path_to_news, path_to_save_results, use_finBERT=True, use_finBERT_pro=True, use_vader=True, use_finvader=True, use_hiv4=True, use_lmd=True, use_distil_roberta=True, use_financialBERT=True, use_sigma=True, use_twit_roberta_base=True, use_twit_roberta_large=True, use_fin_roberta=True, use_finllama=True)
    ```

2. **Run Analysis**:
    ```python
    sentiment_analysis.run_analysis()
    ```