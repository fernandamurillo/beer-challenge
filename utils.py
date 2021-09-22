import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Union, Tuple
from sklearn import preprocessing
from textblob import TextBlob
import re
import string
import nltk


def load_data(data_file_path: str= ('./data/BeerDataScienceProject.csv')) -> pd.DataFrame:
    """Load beer reviews data and drop missing values

    Args: 
        data_file_path (str, optional): the path to the data file. Defaults to './data/BeerDataScienceProject.csv'.
    
    Returns:  
        pd.DataFrame: dataframe of beer review data
    
    """
    # load data and drop missing values
    data = pd.read_csv(data_file_path)
    data = data.dropna()

    return data


def get_rating_features_labels(data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Get feature and label data for user (binary) rating classification using taste, aroma, appearance, and palette for
    features and review_binary for labels

    Args:
        data (pd.DataFrame): dataframe containing full dataset (without missing values)
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: a feature and label dataframe
    
    """
    # Get feature data (already scaled)
    X = data[['review_taste', 'review_aroma', 'review_appearance','review_palette']]

    # Encode (float) labels for classification
    y = data['review_binary'].values
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    return X, y


def add_ratings_binary(data: pd.DataFrame) -> pd.DataFrame:
    """Add binary ratings by turning original rating data (9 classes) into binary labels, rating under 3.5 and under to 0 (bad) and
    ratings over 3.5 to 5 (good) to simplify classification problem for equal class distribution
    
    Args: 
        data (pd.DataFrame): dataframe containing original 9 class ratings data
    
    Returns:
         pd.DataFrame: dataframe with added simplified rating classes 

    """
    data.loc[data['review_overall']<= 3.5,'review_binary'] = 0
    data.loc[data['review_overall']> 3.5,'review_binary'] = 1

    return data
    

def add_review_polarity_subjectivity(data: pd.DataFrame) -> pd.DataFrame:
    """Add written review polarity and subjectivity from TextBlob sentiment analysis

    Args: 
        data (pd.DataFrame): dataframe containing original written reviews
    
    Returns:
         pd.DataFrame: dataframe with added written review polarity and subjectivity
    """
    # Get review polarity and subjectivity using TextBlob sentiment analysis
    pol = lambda x: TextBlob(x).sentiment.polarity
    sub = lambda x: TextBlob(x).sentiment.subjectivity

    # Add polarity and subjectivity columns to data
    data['review_polarity'] = data['review_text'].apply(pol)
    data['review_subjectivity'] = data['review_text'].apply(sub)

    return data


def preprocess_text(text: pd.Series) -> pd.Series:
    """Preprocess and clean text data for Doc2Vec Model, including lowercaseing,
    removing punuctuation and stopwords, tokenize words and lemmatize tokens

    Args:
        text (pd.Series): series containing review sentences as rows
    
    Returns:
        pd.Series: series containing cleaned and lemmatized text for Doc2Vec model

    """
    stopwords = nltk.corpus.stopwords.words('english')
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    # make lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]','',text)
    # remove whitespaces
    text = text.strip()
    # tokenize words
    tokens = nltk.tokenize.word_tokenize(text)
    # remove stop words and lemmatize
    result = [wordnet_lemmatizer.lemmatize(i) for i in tokens if not i in stopwords]
    
    return result
