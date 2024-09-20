# QSIURP Project

## Overview

This project is designed to analyze stock price data and sentiment data to identify and rank pairs of stocks based on various metrics. The analysis involves multiple steps, including data cleaning, augmentation with technical indicators, sentiment analysis using various lexicons and language models, and optimization of trading strategies.

## Project Structure

```
    my_project/
    ├── data/                          # Directory for data files
    │   ├── sentiment_raw_data.csv
    │   ├── sent_pairs.csv
    │   ├── refinitiv_raw_prices.csv
    │   ├── refinitiv_clean_data.csv
    │   └── ...                        
    ├── notebooks/                     # Directory for Jupyter notebooks
    │   ├── Z_final_study.ipynb
    │   ├── find_sentiment_pairs.ipynb
    │   └── ...                        
    ├── src/                           # Source code
    │   ├── __init__.py
    │   ├── sentiment_analysis.py
    │   ├── lexicon_correlation.py
    │   ├── 9_PAIR_RANKER.py
    │   ├── 8_OPTIMIZER.py
    │   ├── 7_PAIRS_BACKTESTER.py
    │   ├── 6_ZSCORE_STRATEGY.py
    │   ├── 5_DATA_MANIPULATOR.py
    │   ├── 4_PAIR_COMPARE.py
    │   ├── 3_PAIR_FINDER.py
    │   ├── 2_INDICATORS.py
    │   ├── 1_STOCK_PRICES.py
    │   ├── 0_helper_functions.py
    │   └── ...                        
    ├── tests/                         # Directory for tests
    │   └── test_sentiment_analysis.py 
    ├── requirements.txt               # List of dependencies
    ├── setup.py                       
    └── README.md                      
```

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/username/QSIURP_project.git
    cd QSIURP_project
    ```

2. Create and activate a virtual environment:

    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Install the package:

    ```sh
    pip install -e .
    ```

## Usage

After installing the package, you can run the project by executing the command:

```sh
my_project
```

This command will execute the main function in the src/main.py file, which contains the primary logic for running the analysis.

## Project Steps

1. **Clean Stock Price Data**
    - Collect daily stock price data of the top 300 companies by market cap from 2023-03-01 to 2024-03-01 using the `0.1_refiniv_get_prices.py` script.
    - Clean the data using the `STOCK_PRICES` module by:
        - Counting the number of data rows for each company's ticker.
        - Identifying the most common number of rows per ticker.
        - Keeping only the tickers that have this most common number of rows.
        - Dropping any tickers with missing data in any of their columns.

2. **Augmenting Data with Technical Indicators & Importing Sentiment Data**
    - Create two data frames to run a correlation study between tickers:
        - **Quantitative Data**: Includes stock price data and various technical indicators using the `INDICATORS` module.
        - **Sentiment Data**: Contains sentiment data collected from 13 different lexicons from 2023-01-01 to 2024-01-01.

3. **Sentiment Analysis**
    - Run sentiment analysis using the `SentimentAnalyzer` class and integrate 13 LLMs and lexicons to analyze sentiment.

4. **Compare Pairs from Quantitative and Sentiment Analysis**
    - Use the `PAIR_COMPARE` module to find the union, intersection, and exclusive pairs in each study.

5. **Rank Pairs for the Period: 2023-03-01 to 2024-03-01**
    - Rank all identified pairs using the `PAIR_RANKER` module based on metrics such as ADF Test Value, P-Value, Half Life, Mean Reversion Significance Level, and 0-Line Crossings.

6. **Optimize Top 100 Pairs**
    - Optimize the top 100 pairs (25 per group) for a Z-Score Strategy during the testing period using the `ZSCORE_STRATEGY`, `PAIR_BACKTESTER`, and `OPTIMIZER` modules.
