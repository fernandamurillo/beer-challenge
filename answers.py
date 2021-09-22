
import pandas as pd
import numpy as np
import xgboost
from sklearn import model_selection, metrics, preprocessing
from gensim.models import doc2vec
import matplotlib.pyplot as plt
from sklearn import manifold

import utils


def top_breweries_strongest_beers(data: pd.DataFrame):
    """Rank 3 breweries which produce strongest beers

    Args:
        data (pd.DataFrame): dataframe containing beer review data
    
    """
    # Sort beers ABV in descending order
    strongest_beers = data.beer_ABV.sort_values(ascending=False)
    # Get top 3 unique strongest beers
    top_3_strongest_beers = strongest_beers.unique()[0:3]
    
    # index by selecting beer id with strongest ABV
    strongest_brewer_id = data.beer_brewerId.loc[data['beer_ABV'].isin(top_3_strongest_beers)]

    print(f"The top 3 brewers with the strongest beers are {set(strongest_brewer_id)}")


def highest_ratings_year(data: pd.DataFrame):
    """Which year did beers enjoy the highest ratings?

    Args:
        data (pd.DataFrame): dataframe containing beer review data
    
    """
    # Convert time integer into datatime type and then to year only
    data['review_year'] = pd.to_datetime(data['review_time'], unit='s').dt.year
    # Find all rows with highest rating (5)
    highest_ratings = data[['review_overall', 'review_year']].loc[data.review_overall == 5]
    # Find year with highest count of 5 star reviews
    highest_year =highest_ratings.value_counts().reset_index().review_year.values[0]

    print(f"The year with highest ratings is {highest_year}")


def important_factors_based_on_ratings(data: pd.DataFrame) -> np.ndarray:
    """Get important features from xgboost classification results

    Args:   
        data (pd.DataFrame): dataframe containing data for classification

    Returns:
        np.ndarray: array containing feature ipmportance from trained xgboost model
    
    """
    # Turn labels into binary classification for equal class distribution
    data = utils.add_ratings_binary(data)
    # Get feature and label data for classifcation from original dataset
    X, y = utils.get_rating_features_labels(data)

    # Grab features from feature matrix
    features = X.columns

    # split data into train and test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X.values, y, test_size=0.2)    

    # Instantiate and train xgboost model for rating classfication
    xgb_model = xgboost.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Grab feature importance scores from trained model
    feature_importance = xgb_model.feature_importances_
    # Find indices of top 2 important features
    top_important_features_ind = np.argpartition(feature_importance, -2)[-2:]

    print(f"The top 2 important features are {features[top_important_features_ind]}")

    return feature_importance


def beer_reccomendations(data: pd.DataFrame):
    """Get top 3 beer (ID) reccomendations beer style based on written reviews using sentiment analysis
    and top ratings(review overall)

    Args:   
        data (pd.DataFrame): dataframe containing data for classification
    
    """
    # Add written review polarity and subjectivity using TextBlob sentiment analysis
    data = utils.add_review_polarity_subjectivity(data)

    # Get best beeres by indexing beer ID with top review polarity and review overall
    best_beers = data['beer_beerId'].loc[ (data['review_polarity'] >= 0.85) & (data['review_overall']==5) ]

    print(f"These three beer reccomendations have 5 star reviews and top positive scores based on written reviews: {best_beers[0:3]}")


def favorite_beer_based_on_written_reviews(data: pd.DataFrame):
    """Get favorite beer style based on written reviews using sentiment analysis

    Args:   
        data (pd.DataFrame): dataframe containing data for classification
    
    """
    # Add written review polarity and subjectivity using TextBlob sentiment analysis
    data = utils.add_review_polarity_subjectivity(data)

    # Get top beer styles by selecting reviews with polarity >= 0.65
    top_styles = data['beer_style'].loc[data['revew_polarity'] >= 0.65].value_counts()

    print(f"The favorite beer style based on written reviews is {top_styles.index[0]}")


def compare_written_review_with_overall_review_score(data: pd.DataFrame):
    """How does written review compare to overall review score for the beer styles? Compare top beer styles using
    written review using sentiment analysis polarity with top beer styles using overall review score 

    Args:   
        data (pd.DataFrame): dataframe containing data for classification
    
    """
    # Add written review polarity and subjectivity using TextBlob sentiment analysis
    data = utils.add_review_polarity_subjectivity(data)

    # Find top beer styles with most number of positive written review polarity (positivity) > 0.65
    top_written_styles = data['beer_style'].loc[data['revew_polarity'] >= 0.65].value_counts()
    top_5_written = top_written_styles[0:5]

    # Find top beer styles with most number of 5 star reviews
    highest_ratings = data[['review_overall', 'beer_style']].loc[data.review_overall == 5]
    top_highest_ratings = highest_ratings.value_counts().reset_index()
    top_5_ratings = top_highest_ratings.beer_style[0:5]

    # Find which beer styles the two methods have in common
    in_common = [x for x in top_5_written if x in top_5_ratings]

    print(f"Favorite beer styles based on written reviews and overall score have {len(in_common)} styles in common. They are: {in_common}")


def similar_beer_drinkers_from_written_reviews(review_text: pd.Series) -> np.ndarray:
    """Find similar beer drinkers from written reviews by using Doc2Vec sentence embedding
    and reducing dimensions of emedding space to visualize clusters (similar users)

    Args: 
        review_text (pd.Series): series containing review sentences as rows

    Returns:
        np.ndarray: Doc2Vec sentence embeddings (document vectors)
    
    """
    # Preprocess text data for Doc2Vec model
    cleaned_text = utils.preprocess_text(review_text)
    # Convert tokenized text data into gensim formated tagged data
    texts = [doc2vec.TaggedDocument(
             words=[word for word in review],
             tags=[i]
         ) for i, review in enumerate(cleaned_text)]

    # create Doc2Vec model
    model = doc2vec.Doc2Vec(vector_size=5,
                alpha=0.025, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)

    # build vocabulary
    model.build_vocab(texts)

    # Get document vectors
    doc_vecs = model.dv.vectors

    return doc_vecs


def vis_similar_users(doc_vectors: np.ndarray):
    """Visualize similar beer drinkers using written reviews sentence embeddings
    and reducing embedded space to two dimensions using tSNE to visualize data

    Args:
        doc_vecs (np.ndarray): Doc2Vec sentence embeddings (document vectors)
    
    """
    # reduce sentence embedding space by tSNE using subset (5,000 points) of data
    tsne = manifold.TSNE(n_components=2, random_state=1, perplexity=30)
    Y = tsne.fit_transform(doc_vectors[0:5_000,:])

    # plot tSNE projections of Doc2Vec sentence emeddings to visualize similar users
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.scatter(Y[:,0], Y[:,1], cmap='winter')
    plt.title('t-SNE projections on Doc2Vec Sentence Embeddings')
    plt.show()


if __name__ == "__main__":
    # Load data
    data = utils.load_data()
    # Print answers from answers.py, Q1
    top_breweries_strongest_beers(data)
    # Q2
    highest_ratings_year(data)
    # Q3
    important_factors_based_on_ratings(data)
    # Q4
    beer_reccomendations(data)
    # Q5
    favorite_beer_based_on_written_reviews(data)
    # Q6
    compare_written_review_with_overall_review_score(data)
    # Q7
    # Get Doc2Vec sentence embeddings from written reviews
    doc_vecs = similar_beer_drinkers_from_written_reviews(data['review_text'])
    # Visualize clusters using t-SNE
    vis_similar_users(doc_vecs)
