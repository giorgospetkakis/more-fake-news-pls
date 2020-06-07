import os
import random
import string

import numpy as np
import pandas as pd
import spacy
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import data
import features
import preprocessing
from utils import go_to_project_root

nlp = spacy.load("en_core_web_md")

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
    for i, a in enumerate(Sub_Authors):
        _ret[a.author_id + f"_sub{i+1}"] = a
    return _ret

# First we take a .csv file with the author IDs and their truth values

go_to_project_root()

all_authors = data.get_processed_data()

df = pd.read_csv("data/IDs_names.csv").to_numpy()
X = df[:,0]
y = df[:,1].astype(int)

PIPELINE_PATH = "data/processed/800s/"

kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=69)

#Start counting so we know in which fold we are in
k = 1
for train_index, test_index in kf.split(X,y):

        
    THIS_PIPELINE_PATH = PIPELINE_PATH + "K{}/".format(k)

    if not os.path.exists(THIS_PIPELINE_PATH):
        os.makedirs(THIS_PIPELINE_PATH)
        print(f"Created directory: {THIS_PIPELINE_PATH}")
  
    y_test = y[test_index]

    # Save y_test values
    pd.DataFrame(y_test).to_csv(THIS_PIPELINE_PATH+"y_test.csv")

    print("Beginning test time augmentation")
    
    # First get all of the test authors
    TestAuthors = {}
    for a in [all_authors[_id] for _id in X[test_index]]:
        TestAuthors[a.author_id] = a
    
    # Now to augment the test data
    THIS_PIPELINE_PATH +=  f"X_test/"
    # Go through each test data point once at a time
    print("Extracting test author data.")
    for author in X[test_index]:
        Test3s_Authors = test_time_augmentation(TestAuthors[author])
    
        # We now have turned one test author into a dictionary of three authors.

        # # First extract the nonlinguistic features
        # Test3s_Authors = features.extract_nonlinguistic_features(Test3s_Authors)

        # Extract semantic similarity
        Test3s_Authors = features.extract_semantic_similarity(Test3s_Authors, model=nlp)

        # # Get the lemmas
        # Test3s_Authors = features.extract_clean_tweets(Test3s_Authors, model=nlp)

        # # Get the nosw
        # Test3s_Authors = features.extract_nosw(Test3s_Authors)

        # # # Lexical features -- TTR requires lemmas
        # Test3s_Authors = features.extract_lexical_features(Test3s_Authors)

        # # Get Named Entities
        # Test3s_Authors = features.extract_named_entities(Test3s_Authors, model=nlp)

        # # Cluster the Named Entities
        # Test3s_Authors = features.extract_mcts_ner(Test3s_Authors, c=ner_clusters) ## NEW FUNCTION HERE

        # # Get POS tags
        # Test3s_Authors = features.extract_pos_tags(Test3s_Authors, model=nlp)

        # # Count POSes and get adjectives
        # Test3s_Authors = features.extract_POS_features(Test3s_Authors, model=nlp)

        # # Cluster the adjectives
        # Test3s_Authors = features.extract_mcts_adj(Test3s_Authors, c=adj_clusters) ## NEW FUNCTION HERE

        # # Extract emotions
        Test3s_Authors = features.extract_emotion_features(Test3s_Authors)

        # # Extract word embeddings
        # Test3s_Authors, _ = features.extract_word_embeddings(Test3s_Authors, c=embeddings)

        test_df = preprocessing.convert_to_df(Test3s_Authors)
        test_df = test_df.drop('author_id', axis=1).to_numpy()
        X_test = test_df[:,:-1]
        
        

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
