import os
import data
import random
import features
import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import pandas as pd
import numpy as np
from utils import go_to_project_root
import spacy

nlp = spacy.load("en_core_web_md")

def train_time_augmentation(ids):
    all_authors = data.get_raw_data()
    _authors = [all_authors[_id] for _id in ids]
    authors = {}

    for a in _authors:
        authors[a.author_id] = a

    ones = [author for author in authors.values() if author.truth == 1]
    zeros = [author for author in authors.values() if author.truth == 0]

    tweets_1 = []
    tweets_0 = []

    for z in zeros:
        for tweet in z.tweets:
            tweets_0 += [tweet]
    random.shuffle(tweets_0)

    for o in ones:
        for tweet in o.tweets:
            tweets_1 += [tweet]
    random.shuffle(tweets_1)

    for author in zeros:
        authors[author.author_id].tweets = []
        for i in range(100):
            authors[author.author_id].tweets += [tweets_0.pop(0)]

    for author in ones:
        authors[author.author_id].tweets = []
        for i in range(100):
            authors[author.author_id].tweets += [tweets_1.pop(0)]

    for i, author in enumerate(authors.keys()):
        authors[author].author_id = f"shuffled-{i + 1}"
        
    return authors


def test_time_augmentation(TestAuthor, n=3):
    # Get in an author object
    # Return a list of author objects created from a subset of tweets
    Sub_Authors = [0] * n
    for i in range(n):
        # Randomly shuffle the tweets of the author
        random.shuffle(TestAuthor.tweets)

        #Save a new author object with half of the tweets
        Sub_Authors[i] = data.Author(TestAuthor.author_id, TestAuthor.tweets[:50], TestAuthor.truth)
        
    _ret = {}
    for a in Sub_Authors:
        _ret[a.author_id] = a
    return _ret

# First we take a .csv file with the author IDs and their truth values

go_to_project_root()

all_authors = data.get_processed_data()

df = pd.read_csv("data/IDs_names.csv").to_numpy()
X = df[:,0]
y = df[:,1].astype(int)

PIPELINE_PATH = "data/processed/"

kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=69)

#Start counting so we know in which fold we are in
k = 1
for train_index, test_index in kf.split(X,y):
    print("Beginning k fold {}".format(k))
    
    ############################ TRAINING ##############################
    
    Train_Authors = {}
    for a in [all_authors[_id] for _id in X[train_index]]:
        Train_Authors[a.author_id] = a

    y_train = y[train_index]
    
    print("Augmenting training data.")
    
    # Augment training data. Then extract the features for it
    augmentations = train_time_augmentation(X[train_index])
    
    # # First extract the nonlinguistic features
    augmentations = features.extract_nonlinguistic_features(augmentations)

    # # Extract semantic similarity
    augmentations = features.extract_semantic_similarity(augmentations, model=nlp)

    # # Get the lemmas
    augmentations = features.extract_clean_tweets(augmentations, model=nlp)

    # # Lexical features -- TTR requires lemmas
    augmentations = features.extract_lexical_features(augmentations)

    # # Get Named Entities
    augmentations = features.extract_named_entities(augmentations, model=nlp)

    # # Get POS tags
    augmentations = features.extract_pos_tags(augmentations, model=nlp)

    # # Count POSes and get adjectives
    augmentations = features.extract_POS_features(augmentations, model=nlp)

    # # Extract emotions
    augmentations = features.extract_emotion_features(augmentations)
    
    ################# FEATURES ALL EXTRACTED ############
    
    print("Features have been extracted. Now clustering.")
    
    Train_Authors.update(augmentations)
    
    # Cluster the Named Entities
    Train_Authors, ner_clusters = features.extract_mcts_ner(Train_Authors)
    
    # Cluster the adjectives
    Train_Authors, adj_clusters = features.extract_mcts_adj(Train_Authors)

    # Create dataframe of what
    train_df = preprocessing.convert_to_df(Train_Authors)
    
    
    train_df = train_df.drop('author_id', axis=1).to_numpy()
    X_train = train_df[:,:-1]
    
    THIS_PIPELINE_PATH = PIPELINE_PATH + "K{}/".format(k)

    if not os.path.exists(THIS_PIPELINE_PATH):
        os.makedirs(THIS_PIPELINE_PATH)
        print(f"Created directory: {THIS_PIPELINE_PATH}")
    
    print("Saving training data.")
    
    pd.DataFrame(X_train).to_csv(THIS_PIPELINE_PATH+"X_train.csv")
    pd.DataFrame(y_train).to_csv(THIS_PIPELINE_PATH+"y_train.csv")
    
    # Write clusters (if you want to further modularize this. you can save the test and train indices and separate these two parts of the feature extraction
    with open(THIS_PIPELINE_PATH + 'ner_clusters.txt', 'w', encoding='utf-8') as f:
        for item in ner_clusters:
            f.write("%s\n" % item)
        
    with open(THIS_PIPELINE_PATH + 'adj_clusters.txt', 'w', encoding='utf-8') as f:
        for item in adj_clusters:
            f.write("%s\n" % item)
        
    ############################ TESTING DATA ##############################
    
    y_test = y[test_index]
    # preds = []
    
    # Save y_test values
    pd.DataFrame(y_test).to_csv(THIS_PIPELINE_PATH+"y_test.csv")

    print("Beginning test time augmentation")
    
    # First get all of the test authors
    TestAuthors = {}
    for a in [all_authors[_id] for _id in X[test_index]]:
        TestAuthors[a.author_id] = a
    
    # Now to augment the test data
    
    # Go through each test data point once at a time
    print("Extracting test author data.")
    for author in X[test_index]:
        Test3s_Authors = test_time_augmentation(TestAuthors[author])
    
        # We now have turned one test author into a dictionary of three authors.

        # First extract the nonlinguistic features
        Test3s_Authors = features.extract_nonlinguistic_features(Test3s_Authors)

        # Extract semantic similarity
        Test3s_Authors = features.extract_semantic_similarity(Test3s_Authors, model=nlp)

        # Get the lemmas
        Test3s_Authors = features.extract_clean_tweets(Test3s_Authors, model=nlp)

        # Lexical features -- TTR requires lemmas
        Test3s_Authors = features.extract_lexical_features(Test3s_Authors)

        # Get Named Entities
        Test3s_Authors = features.extract_named_entities(Test3s_Authors, model=nlp)

        # Cluster the Named Entities
        Test3s_Authors = features.extract_mcts_ner(Test3s_Authors, c=ner_clusters) ## NEW FUNCTION HERE

        # Get POS tags
        Test3s_Authors = features.extract_pos_tags(Test3s_Authors, model=nlp)

        # Count POSes and get adjectives
        Test3s_Authors = features.extract_POS_features(Test3s_Authors, model=nlp)

        # Cluster the adjectives
        Test3s_Authors = features.extract_mcts_adj(Test3s_Authors, c=adj_clusters) ## NEW FUNCTION HERE

        # Extract emotions
        Test3s_Authors = features.extract_emotion_features(Test3s_Authors)

        test_df = preprocessing.convert_to_df(Test3s_Authors)
        test_df = test_df.drop('author_id', axis=1).to_numpy()
        X_test = test_df[:,:-1]
        
        THIS_PIPELINE_PATH = THIS_PIPELINE_PATH + "X_test/"

        if not os.path.exists(THIS_PIPELINE_PATH):
            os.makedirs(THIS_PIPELINE_PATH)
            print(f"Created directory: {THIS_PIPELINE_PATH}")
        
        # Save all three datapoints corresponding to this author in a CSV file in a folder called "X_test"
        pd.DataFrame(X_test).to_csv(THIS_PIPELINE_PATH+f"{author}.csv")
        print("|",end="")
        
    print()
    print(f"Batch {k} finished")
    print()
    print()
    k += 1
    

