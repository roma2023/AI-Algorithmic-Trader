# src/__init__.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing my_project package")

from .sentiment_analysis import SentimentAnalysis
from .lexicon_correlation import LexiconCorrelation
from _9_PAIR_RANKER import PAIR_RANKER
from _8_OPTIMIZER import OPTIMIZER
from ._7_PAIRS_BACKTESTER import PAIRS_BACKTESTER
from ._6_ZSCORE_STRATEGY import ZSCORE_STRATEGY
from ._5_DATA_MANIPULATOR import DATA_MANIPULATOR
from ._4_PAIR_COMPARE import PAIR_COMPARE
from ._3_PAIR_FINDER import PAIR_FINDER
from ._2_INDICATORS import INDICATORS
from ._1_STOCK_PRICES import STOCK_PRICES