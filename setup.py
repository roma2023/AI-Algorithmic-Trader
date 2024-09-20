from setuptools import setup, find_packages

setup(
    name='QSIURP_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Basic Data Science
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',

        # Sentiment analysis
        'scikit-learn',
        'sklearn_preprocessing',
        'nltk',
        'transformers',
        'torch',
        'finvader', 
        'pysentiment2', 
        'scipy',
        'openpyxl',

        # Technical Analyis
        'ta',
        'yfinance',
        'statsmodels'
    ],
    entry_points={
        'console_scripts': [
            'my_project=main:main',  # Adjust accordingly
        ],
    },
)
