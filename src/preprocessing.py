## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

import data
from utils import go_to_project_root
import numpy as np
import pandas as pd

filepath = 'data/processed/'

def convert_to_df(authors, export=False):
    '''
    Converts the given authors to a pandas DataFrame.
    Parameters:
        authors (Author dict):
            The authors to be converted.
        export (Boolean):
            Whether the function will export to file.
            False by default
    Returns:
        pandas DataFrame
    '''
    # Can't not hard-code this
    # Create table, fill table, convert to dataframe, name columns, return
    table = np.zeros((len(list(authors.values())), 47))

    for i, a in enumerate(list(authors.values())):
        table[i] = [
            # Semantic Similarity
            a.max_similar, 
            a.min_similar, 
            a.mean_similar, 
            a.number_identical, 

            # NER / Clustring
            a.most_common_ner_score,
            a.most_common_adj_score,
            
            # Non-Linguistic
            a.nonlinguistic_features['url_max'],
            a.nonlinguistic_features['url_mean'],
            a.nonlinguistic_features['hashtag_max'],
            a.nonlinguistic_features['hashtag_mean'],
            a.nonlinguistic_features['user_max'],
            a.nonlinguistic_features['user_mean'],
            a.nonlinguistic_features['emoji_mean'],
            a.nonlinguistic_features['emoji_max'],
            a.nonlinguistic_features['exclamation_mean'],
            a.nonlinguistic_features['exclamation_max'],
            a.nonlinguistic_features['period_mean'],
            a.nonlinguistic_features['period_max'],
            a.nonlinguistic_features['question_mean'],
            a.nonlinguistic_features['question_max'],
            a.nonlinguistic_features['comma_mean'],
            a.nonlinguistic_features['comma_max'],
            a.nonlinguistic_features['allcaps_ratio'],
            a.nonlinguistic_features['allcaps_inclusion_ratio'],
            a.nonlinguistic_features['titlecase_ratio'],
            a.nonlinguistic_features['mean_words'],
            a.nonlinguistic_features['retweet_percentage'],

            # POS Tags
            a.POS_counts['ADJ_mean'],
            a.POS_counts['ADP_mean'],
            a.POS_counts['ADV_mean'],
            a.POS_counts['AUX_mean'],
            a.POS_counts['CONJ_mean'],
            a.POS_counts['CCONJ_mean'],
            a.POS_counts['DET_mean'],
            a.POS_counts['INTJ_mean'],
            a.POS_counts['NOUN_mean'],
            a.POS_counts['NUM_mean'],
            a.POS_counts['PART_mean'],
            a.POS_counts['PRON_mean'],
            a.POS_counts['PROPN_mean'],
            a.POS_counts['PUNCT_mean'],
            a.POS_counts['SCONJ_mean'],
            a.POS_counts['SYM_mean'],
            a.POS_counts['VERB_mean'],
            a.POS_counts['X_mean'],
            a.POS_counts['TOKEN_mean'],

            # https://www.youtube.com/watch?v=DpxDl68brww
            a.truth
            ]

    df = pd.DataFrame(table, columns=[
        "max_similar", 
        "min_similar", 
        "mean_similar", 
        "number_identical", 
        "mcts_ner",
        "mcts_adj",
        'url_max',
        'url_mean',
        'hashtag_max',
        'hashtag_mean',
        'user_max',
        'user_mean',
        'emoji_mean',
        'emoji_max',
        'exclamation_mean',
        'exclamation_max',
        'period_mean',
        'period_max',
        'question_mean',
        'question_max',
        'comma_mean',
        'comma_max',
        'allcaps_ratio',
        'allcaps_inclusion_ratio',
        'titlecase_ratio',
        'mean_words',
        'retweet_percentage',
        'ADJ',
        'ADP', 
        'ADV', 
        'AUX', 
        'CONJ', 
        'CCONJ', 
        'DET', 
        'INTJ', 
        'NOUN', 
        'NUM',
        'PART',
        'PRON',
        'PROPN',
        'PUNCT',
        'SCONJ',
        'SYM',
        'VERB',
        'X',
        'TOKEN',
        "truth"
        ])

    # Enable to export to file
    if export:
        go_to_project_root()
        df.to_csv(filepath+"processed.csv")

    return df
