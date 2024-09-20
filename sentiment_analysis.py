import numpy as np
import pandas as pd
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from finvader import finvader
import pysentiment2 as ps
import urllib.request
import csv

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Remove Unwanted Warnings
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

ssl._create_default_https_context = ssl._create_unverified_context

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer


"""
                                    << SentimentAnalysis (module) >>

Class Initialization Inputs:
    (1) path_to_news                : str  | Path to the news data file.
    (2) path_to_save_results        : str  | Path to save the analysis results.
    (3) use_finBERT                 : bool | Whether to use FinBERT for sentiment analysis (default: True).
    (4) use_finBERT_pro             : bool | Whether to use FinBERT-pro for sentiment analysis (default: True).
    (5) use_vader                   : bool | Whether to use VADER for sentiment analysis (default: True).
    (6) use_finvader                : bool | Whether to use finVADER for sentiment analysis (default: True).
    (7) use_hiv4                    : bool | Whether to use HIV4 for sentiment analysis (default: True).
    (8) use_lmd                     : bool | Whether to use LMD for sentiment analysis (default: True).
    (9) use_distil_roberta          : bool | Whether to use DistilRoBERTa for sentiment analysis (default: True).
    (10) use_financialBERT          : bool | Whether to use FinancialBERT for sentiment analysis (default: True).
    (11) use_sigma                  : bool | Whether to use Sigma for sentiment analysis (default: True).
    (12) use_twit_roberta_base      : bool | Whether to use Twitter-roBERTa-base for sentiment analysis (default: True).
    (13) use_twit_roberta_large     : bool | Whether to use Twitter-roBERTa-large for sentiment analysis (default: True).
    (14) use_fin_roberta            : bool | Whether to use Financial-RoBERTa for sentiment analysis (default: True).
    (15) use_finllama               : bool | Whether to use FinLLama for sentiment analysis (default: True).
    (16) batch_size                 : int  | Batch size for processing headlines (default: 64).

------------ User-facing Methods --------------
    (1) run_analysis():
        => Runs sentiment analysis using the specified models. Saves the results to a CSV file.

        => Inputs: None
        => Outputs: None

    (2) get_daily_sentiment():
        => Aggregates daily sentiment scores for each ticker and saves the results to a CSV file.

        => Inputs: None
        => Outputs: None

---------------- Usage ------------------------
sentiment_analysis = SentimentAnalysis(
    path_to_news='path/to/your/news.csv',
    path_to_save_results='path/to/save/results.csv',
    use_finBERT=True,
    use_finBERT_pro=True,
    use_vader=True,
    use_finvader=True,
    use_hiv4=True,
    use_lmd=True,
    use_distil_roberta=True,
    use_financialBERT=True,
    use_sigma=True,
    use_twit_roberta_base=True,
    use_twit_roberta_large=True,
    use_fin_roberta=True,
    use_finllama=True
)
sentiment_analysis.run_analysis()
"""

class SentimentAnalysis():
    def __init__(self, path_to_news: str, 
                 path_to_save_results: str,
                 use_finBERT: bool = True,
                 use_finBERT_pro: bool = True, 
                 use_vader: bool = True, 
                 use_finvader: bool = True,
                 use_hiv4: bool = True, 
                 use_lmd: bool = True,
                 use_distil_roberta: bool = True,
                 use_financialBERT: bool = True,
                 use_sigma: bool = True,
                 use_twit_roberta_base: bool = True,
                 use_twit_roberta_large: bool = True,
                 use_fin_roberta: bool = True,
                 use_finllama: bool = True,
                 batch_size: int = 64):
        # Initialize parameters into class
        self.path_to_news = path_to_news
        self.path_to_save_results = path_to_save_results
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print("Running on GPU")
        else:
            print("Running on CPU")

        # Model flags
        self.use_finBERT = use_finBERT
        self.use_finBERT_pro = use_finBERT_pro
        self.use_vader = use_vader
        self.use_finvader = use_finvader
        self.use_hiv4 = use_hiv4
        self.use_lmd = use_lmd
        self.use_distil_roberta = use_distil_roberta
        self.use_financialBERT = use_financialBERT
        self.use_sigma = use_sigma
        self.use_twit_roberta_base = use_twit_roberta_base
        self.use_twit_roberta_large = use_twit_roberta_large
        self.use_fin_roberta = use_fin_roberta
        self.use_finllama = use_finllama

        # Model and tokenizer placeholders
        self.finBERT_model = None
        self.finBERT_tokenizer = None
        self.finBERT_pro_model = None
        self.finBERT_pro_tokenizer = None
        self.vader_analyzer = None
        self.finvader_analyzer = None
        self.hiv4_analyzer = None
        self.lmd_analyzer = None
        self.distil_roberta_model = None
        self.distil_roberta_tokenizer = None
        self.financialBERT_model = None
        self.financialBERT_tokenizer = None
        self.sigma_model = None
        self.sigma_tokenizer = None
        self.twit_roberta_base_model = None
        self.twit_roberta_base_tokenizer = None
        self.twit_roberta_large_model = None
        self.twit_roberta_large_tokenizer = None
        self.fin_roberta_model = None
        self.fin_roberta_tokenizer = None
        self.finllama_model = None
        self.finllama_tokenizer = None
        self.models_initialized = False

    """

    ------------ Backend Methods --------------
        (1) initialize_models():
            => Initializes all the models and tokenizers as per the flags set during initialization.

            => Inputs: None
            => Outputs: None
    """
    def initialize_models(self):
        # Initialize all models and tokenizers here
        if self.use_finBERT and self.finBERT_model is None:
            self.finBERT_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
            self.finBERT_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        
        if self.use_finBERT_pro and self.finBERT_pro_model is None:
            self.finBERT_pro_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.finBERT_pro_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')

        if self.use_vader and self.vader_analyzer is None:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        if self.use_finvader and self.finvader_analyzer is None:
            self.finvader_analyzer = finvader
        
        if self.use_hiv4 and self.hiv4_analyzer is None:
            self.hiv4_analyzer = ps.HIV4()
        
        if self.use_lmd and self.lmd_analyzer is None:
            self.lmd_analyzer = ps.LM()
        
        if self.use_distil_roberta and self.distil_roberta_model is None:
            self.distil_roberta_model = AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')
            self.distil_roberta_tokenizer = AutoTokenizer.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')

        if self.use_financialBERT and self.financialBERT_model is None:
            self.financialBERT_model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis", num_labels=3)
            self.financialBERT_tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
        
        if self.use_sigma and self.sigma_model is None:
            self.sigma_model = AutoModelForSequenceClassification.from_pretrained('Sigma/financial-sentiment-analysis')
            self.sigma_tokenizer = AutoTokenizer.from_pretrained('Sigma/financial-sentiment-analysis')

        if self.use_twit_roberta_base and self.twit_roberta_base_model is None:
            self.twit_roberta_base_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
            self.twit_roberta_base_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        
        if self.use_twit_roberta_large and self.twit_roberta_large_model is None:
            self.twit_roberta_large_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
            self.twit_roberta_large_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
        
        if self.use_fin_roberta and self.fin_roberta_model is None:
            self.fin_roberta_model = AutoModelForSequenceClassification.from_pretrained('soleimanian/financial-roberta-large-sentiment')
            self.fin_roberta_tokenizer = AutoTokenizer.from_pretrained('soleimanian/financial-roberta-large-sentiment')
        
        if self.use_finllama and self.finllama_model is None:
            self.finllama_model = AutoModelForSequenceClassification.from_pretrained('roma2025/FinLlama-3-8B', num_labels=3)
            self.finllama_tokenizer = AutoTokenizer.from_pretrained('roma2025/FinLlama-3-8B')
            # Set pad_token and pad_token_id
            self.finllama_tokenizer.pad_token = self.finllama_tokenizer.eos_token
            self.finllama_model.config.pad_token_id = self.finllama_tokenizer.eos_token_id
        
        self.models_initialized = True
    
    """
    (2) load_data():
        => Loads the news data from the specified path.

        => Inputs: None
        => Outputs: None
    """
    
    def load_data(self):
        # Load news data from the specified path
        self.news_data = pd.read_csv(self.path_to_news)

    """
        (3) process_batch(headlines, tokenizer, model, weights):
        => Processes a batch of headlines, performs sentiment analysis, and calculates weighted scores.

        => Inputs:
            headlines : list  | List of headlines to process.
            tokenizer : object| Tokenizer for the specified model.
            model     : object| Model to use for sentiment analysis.
            weights   : array | Weights for calculating the sentiment score.

        => Outputs:
            list | List of weighted sentiment scores for the batch.
    """

    def process_batch(self, headlines, tokenizer, model, weights):
        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # Move inputs to GPU
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        batch_scores = np.dot(probabilities, weights)
        return np.round(batch_scores, 4).tolist()
    
    """
        (4) finBERT_analysis():
        => Performs sentiment analysis using FinBERT.

        => Inputs: None
        => Outputs:
            list | List of FinBERT sentiment scores.
    """

    def finBERT_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.finBERT_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'neutral' else 1 if label.lower() == 'positive' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.finBERT_tokenizer, self.finBERT_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.finBERT_tokenizer, self.finBERT_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (5) finBERT_pro_analysis():
        => Performs sentiment analysis using FinBERT-pro.

        => Inputs: None
        => Outputs:
            list | List of FinBERT-pro sentiment scores.
    """
    # Add similar methods for other models
    def finBERT_pro_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.finBERT_pro_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'neutral' else 1 if label.lower() == 'positive' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.finBERT_pro_tokenizer, self.finBERT_pro_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.finBERT_pro_tokenizer, self.finBERT_pro_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (6) vader_analysis():
        => Performs sentiment analysis using VADER.

        => Inputs: None
        => Outputs:
            list | List of VADER sentiment scores.
    """
    def vader_analysis(self):
        if not self.models_initialized:
            self.initialize_models()
        
        scores = self.news_data['headline'].apply(self.vader_analyzer.polarity_scores)
        v_neg = scores.apply(lambda x: x['neg'])
        v_neu = scores.apply(lambda x: x['neu'])
        v_pos = scores.apply(lambda x: x['pos'])
        vader_scores = v_neg * (-1) + v_neu * (0) + v_pos * (1)

        return vader_scores.tolist()

    """
        (7) finvader_analysis():
        => Performs sentiment analysis using finVADER.

        => Inputs: None
        => Outputs:
            list | List of finVADER sentiment scores.
    """
    def finvader_analysis(self):
        if not self.models_initialized:
            self.initialize_models()
        
        scores = self.news_data['headline'].apply(lambda x: self.finvader_analyzer(x, use_sentibignomics=True, use_henry=True))
        v_neg = scores.apply(lambda x: x['neg'])
        v_neu = scores.apply(lambda x: x['neu'])
        v_pos = scores.apply(lambda x: x['pos'])
        finvader_scores = v_neg * (-1) + v_neu * (0) + v_pos * (1)

        return finvader_scores.tolist()

    """
        (8) hiv4_analysis():
        => Performs sentiment analysis using HIV4.

        => Inputs: None
        => Outputs:
            list | List of HIV4 sentiment scores.
    """
    def hiv4_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        hiv4_scores = self.news_data['headline'].apply(lambda x: self.hiv4_analyzer.get_score(nltk.word_tokenize(x)))
        polarity_hiv4 = hiv4_scores.apply(lambda x: x['Polarity'])

        return polarity_hiv4.tolist()

    """
        (9) lmd_analysis():
        => Performs sentiment analysis using LMD.

        => Inputs: None
        => Outputs:
            list | List of LMD sentiment scores.
    """
    def lmd_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        lmd_scores = self.news_data['headline'].apply(lambda x: self.lmd_analyzer.get_score(nltk.word_tokenize(x)))
        polarity_lmd = lmd_scores.apply(lambda x: x['Polarity'])

        return polarity_lmd.tolist()
    """
        (10) distil_roberta_analysis():
        => Performs sentiment analysis using DistilRoBERTa.

        => Inputs: None
        => Outputs:
            list | List of DistilRoBERTa sentiment scores.
    """
    def distil_roberta_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.distil_roberta_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'neutral' else 1 if label.lower() == 'positive' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.distil_roberta_tokenizer, self.distil_roberta_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.distil_roberta_tokenizer, self.distil_roberta_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (11) financialBERT_analysis():
        => Performs sentiment analysis using FinancialBERT.

        => Inputs: None
        => Outputs:
            list | List of FinancialBERT sentiment scores.
    """
    def financialBERT_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.financialBERT_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'neutral' else 1 if label.lower() == 'positive' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.financialBERT_tokenizer, self.financialBERT_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.financialBERT_tokenizer, self.financialBERT_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (12) sigma_analysis():
        => Performs sentiment analysis using Sigma.

        => Inputs: None
        => Outputs:
            list | List of Sigma sentiment scores.
    """
    def sigma_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.sigma_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'label_1' else 1 if label.lower() == 'label_2' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.sigma_tokenizer, self.sigma_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.sigma_tokenizer, self.sigma_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (13) twit_roberta_base_analysis():
        => Performs sentiment analysis using Twitter-roBERTa-base.

        => Inputs: None
        => Outputs:
            list | List of Twitter-roBERTa-base sentiment scores.
    """
    def twit_roberta_base_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.twit_roberta_base_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'label_1' else 1 if label.lower() == 'label_2' else -1 for label in label_order])
        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.twit_roberta_base_tokenizer, self.twit_roberta_base_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.twit_roberta_base_tokenizer, self.twit_roberta_base_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (14) twit_roberta_large_analysis():
        => Performs sentiment analysis using Twitter-roBERTa-large.

        => Inputs: None
        => Outputs:
            list | List of Twitter-roBERTa-large sentiment scores.
    """
    def twit_roberta_large_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.twit_roberta_large_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'neutral' else 1 if label.lower() == 'positive' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.twit_roberta_large_tokenizer, self.twit_roberta_large_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.twit_roberta_large_tokenizer, self.twit_roberta_large_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (15) fin_roberta_analysis():
        => Performs sentiment analysis using Financial-RoBERTa.

        => Inputs: None
        => Outputs:
            list | List of Financial-RoBERTa sentiment scores.
    """
    def fin_roberta_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        config = self.fin_roberta_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'neutral' else 1 if label.lower() == 'positive' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(self.news_data)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = self.news_data['headline'][i:i + self.batch_size].tolist()
            batch_scores = self.process_batch(batch_headlines, self.fin_roberta_tokenizer, self.fin_roberta_model, weights)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = self.news_data['headline'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.fin_roberta_tokenizer, self.fin_roberta_model, weights)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(self.news_data):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(self.news_data)})")

        return weighted_scores


    """
        (16) finllama_analysis():
        => Performs sentiment analysis using FinLLama.

        => Inputs: None
        => Outputs:
            list | List of FinLLama sentiment scores.
    """
    def finllama_analysis(self):
        if not self.models_initialized:
            self.initialize_models()

        def format_instruction_following_dataset(df, instructions):
            formatted_data = []
            for idx, row in df.iterrows():
                instruction = instructions[idx % len(instructions)]
                formatted_input = f"{instruction}\n{row['headline']}\nSentiment:"
                formatted_data.append({'input': formatted_input})
            return pd.DataFrame(formatted_data)

        instructions = [
                "Determine the sentiment of the financial news as negative, neutral or positive:",
                "Classify the tone of the financial news as positive, neutral, or negative:",
                "Analyze the sentiment of the financial news as neutral, positive, or negative:",
                "Evaluate the sentiment of the financial news as positive, neutral, or negative:",
                "Identify whether the sentiment of the financial news is negative, neutral, or positive:",
                "Assess the financial news and determine if the sentiment is positive, neutral, or negative:",
                "Judge the sentiment conveyed in the financial news as neutral, negative, or positive:",
                "Review the sentiment of the financial news and classify it as positive, neutral, or negative:",
                "Determine if the financial news sentiment is positive, neutral, or negative:",
                "Classify whether the financial news sentiment is positive, neutral, or negative:"
            ]
        
        # format the input text
        selected_columns = ['headline']

        # Create a new DataFrame with only the selected columns
        new_df = self.news_data[selected_columns]
        formatted_news = format_instruction_following_dataset(new_df, instructions)

        config = self.finllama_model.config
        label_order = [config.id2label[i] for i in range(len(config.id2label))]
        weights = np.array([0 if label.lower() == 'label_1' else 1 if label.lower() == 'label_2' else -1 for label in label_order])

        weighted_scores = []
        num_samples = len(formatted_news)

        # Process headlines in batches
        for i in range(0, num_samples - self.batch_size, self.batch_size):
            batch_headlines = formatted_news['input'][i:i + self.batch_size].tolist()
            if len(batch_headlines) != self.batch_size:
                print(f"Processing batch {i//self.batch_size + 1}: {len(batch_headlines)} headlines")  # Debug statement
            batch_scores = self.process_batch(batch_headlines, self.finllama_tokenizer, self.finllama_model)
            weighted_scores.extend(batch_scores)

        # Process any remaining samples if they exist
        remainder = num_samples % self.batch_size
        if remainder != 0:
            remaining_headlines = formatted_news['input'][-remainder:].tolist()
            remaining_scores = self.process_batch(remaining_headlines, self.finllama_tokenizer, self.finllama_model)
            weighted_scores.extend(remaining_scores)

        # Check if the lengths match
        if len(weighted_scores) != len(formatted_news):
            print(f"Length of values ({len(weighted_scores)}) does not match length of index ({len(formatted_news)})")

        return weighted_scores

    def run_analysis(self):
        # Load data
        self.load_data()

        # Run all analyses based on the flags
        if self.use_finBERT:
            self.news_data['finBERT_score'] = self.finBERT_analysis()
        if self.use_finBERT_pro:
            self.news_data['FinBERT-pro_score'] = self.finBERT_pro_analysis()
        if self.use_vader:
            self.news_data['VADER_score'] = self.vader_analysis()
        if self.use_finvader:
            self.news_data['finVADER_score'] = self.finvader_analysis()
        if self.use_hiv4:
            self.news_data['HIV4_Polarity'] = self.hiv4_analysis()
        if self.use_lmd:
            self.news_data['LMD_Polarity'] = self.lmd_analysis()
        if self.use_distil_roberta:
            self.news_data['distil_score'] = self.distil_roberta_analysis()
        if self.use_financialBERT:
            self.news_data['FinancialBERT_score'] = self.financialBERT_analysis()
        if self.use_sigma:
            self.news_data['Sigma_score'] = self.sigma_analysis()
        if self.use_twit_roberta_base:
            self.news_data['Twitter_roBERTa_score'] = self.twit_roberta_base_analysis()
        if self.use_twit_roberta_large:
            self.news_data['roBERTa_large_score'] = self.twit_roberta_large_analysis()
        if self.use_fin_roberta:
            self.news_data['Financial_RoBERTa_score'] = self.fin_roberta_analysis()
        if self.use_finllama:
            self.news_data['finllama_score'] = self.finllama_analysis()
        
        print("Here is the head of the output table: ")
        print(self.news_data.head())

        # Save the DataFrame to a CSV file
        self.news_data.to_csv(self.path_to_save_results, index=False)
        print(f"Analysis complete and saved to {self.path_to_save_results}")


    def align_dates(self, df): 
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Create a complete date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        complete_date_range = pd.date_range(start=min_date, end=max_date)
        
        # Initialize a list to store dataframes for each ticker
        df_list = []

        # Group by ticker and reindex each group's data
        for ticker, group in df.groupby('tic'):
            group = group.set_index('date').reindex(complete_date_range).reset_index()
            group['tic'] = ticker
            df_list.append(group)

        # Concatenate all the reindexed dataframes
        aligned_df = pd.concat(df_list)

        # Fill missing values as needed (e.g., forward fill or backward fill)
        aligned_df.fillna(method='ffill', inplace=True)
        aligned_df.fillna(method='bfill', inplace=True)

        # Reset the index and rename columns
        aligned_df.rename(columns={'index': 'date'}, inplace=True)
        aligned_df.reset_index(drop=True, inplace=True) 

        return aligned_df

    """
        (18) get_daily_sentiment():
            => Aggregates daily sentiment scores for each ticker and saves the results to a CSV file.

            => Inputs: None
            => Outputs: None
    """
    def get_daily_sentiment(self):
        sentiment_df = pd.read_csv(self.path_to_save_results)
        sentiment_df.rename(columns={'versionCreated': 'date'}, inplace=True)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])  # Ensure this is also datetime

        agg_list = []

        if self.use_finBERT:
            agg_list += [('finBERT_score', 'mean')]
        if self.use_finBERT_pro:
            agg_list += [('FinBERT-pro_score', 'mean')]
        if self.use_vader:
            agg_list += [('VADER_score', 'mean')]
        if self.use_finvader:
            agg_list += [('finVADER_score', 'mean')]
        if self.use_hiv4:
            agg_list += [('HIV4_Polarity', 'mean')]
        if self.use_lmd:
            agg_list += [('LMD_Polarity', 'mean')]
        if self.use_distil_roberta:
            agg_list += [('distil_score', 'mean')]
        if self.use_financialBERT:
            agg_list += [('FinancialBERT_score', 'mean')]
        if self.use_sigma:
            agg_list += [('Sigma_score', 'mean')]
        if self.use_twit_roberta_base:
            agg_list += [('Twitter_roBERTa_score', 'mean')]
        if self.use_twit_roberta_large:
            agg_list += [('roBERTa_large_score', 'mean')]
        if self.use_fin_roberta:
            agg_list += [('Financial_RoBERTa_score', 'mean')]
        if self.use_finllama:
            agg_list += [('finllama_score', 'mean')]


        agg_funcs = dict(agg_list)
        print(agg_funcs)

        sentiment_df['RIC'] = sentiment_df['RIC'].str.split('.').str[0]  # Adjusting RIC to match tic
        sentiment_df = sentiment_df.groupby(['RIC', 'date']).agg(agg_funcs).reset_index()

        # Rename RIC to tic for consistency in merging
        sentiment_df.rename(columns={'RIC': 'tic'}, inplace=True)

        # Align dates for further transpose and correlation 
        sentiment_df = self.align_dates(sentiment_df)

        print("Here is the head of the output table: ")
        print(sentiment_df.head()) 
        new_path_to_save = self.path_to_save_results[:-4] + "_daily.csv"
        sentiment_df.to_csv(new_path_to_save, index=False)
        print(f"Analysis complete and saved to {new_path_to_save}")
