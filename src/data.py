## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

import jsonpickle
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
from utils import go_to_project_root

RAW_DATA_PATH = "data/raw/"
PREPROCESSED_DATA_PATH = "data/processed/"

class Author:
        
    def __init__(self, author_id, tweets, truth):
        self.author_id = author_id
        self.truth = truth
        self.tweets = tweets

        self.ents = []

        ''' 
        The ents variable contains a list of lists.
        The first coordinate in the list refers to the tweet number.
        The nested list contains tuples of the named entities located in the tweet
            To access the entity type (e.g. PERSON), call .label_ (fx. author.ents[0][0].label_ )
            To access the entity name (e.g. Joe Jonas), call .text (fx. author.ents[0][0].text )
        To understand the labels, look here: https://spacy.io/api/annotation
        '''

        self.POS_tags = []

        ''' 
        The POS_tags variable contains a list of list.
        The first coordinate of the list refers to the tweet of the author
        The nested list contains a list of the sequence of POS tags of tweet.
        To understand the tags look here: https://spacy.io/api/annotation (We use the coarse-grained tags)
        '''

        self.clean = []

        ''' 
        The lemma variable contains a list of list. All non-alpha characters have been removed from the text.
        The first coordinate of the list refers to the tweet of the author
        The nested list contains a sequence (separated by spaces) of the lemmas of the words. URL and HASHTAGS notes have been removed.
        '''

        self.similarities = None
        '''
        Contains an array of the similarity measure (between 0 and 1) of two tweets indexed on the two axes.
        Half an array (values are not repeated). Empty values are filled in as -1.
        Saved for now, but perhaps would be wise to conserve only what we need?
        From this row, we save the following features: max similarity, min similarity, number of identical tweets, and mean similarity.
        '''

        self.max_similar = None
        self.min_similar = None
        self.mean_similar = None
        self.number_identical = None

def __get_truth_vals__(directory):
    truth_dict = {}
    for f in listdir(directory):
        if f.split(".")[-1] != "xml":
            truth_file = open(join(directory, f), "r", encoding='utf-8')
            for line in truth_file.readlines():
                split = line.split(":::")
                truth_dict[split[0]] = int(split[1].strip())
            break
    return truth_dict

def __import_from__(directory):
    file_list = [f for f in listdir(directory) if isfile(join(directory, f)) and f.split(".")[-1] == "xml"]
    truth_vals = __get_truth_vals__(directory)
    author_dict = {}
    for file in file_list:
        author_id = file.split(".")[0]
        author_dict[author_id] = Author(author_id, __parse_tweets__(join(directory, file)), truth_vals[author_id])
    return author_dict

def __parse_tweets__(filepath):
    tweets_file = open(filepath, "r", encoding='utf-8')
    content = tweets_file.read()
    soup = BeautifulSoup(content, 'xml')
    tweets = soup.find_all('document')
    return [t.get_text() for t in tweets]

def get_raw_data(lang='en'):
    '''
    Returns the raw data in the selected language.

    Parameters:
        lang (str): 
            The return language. Valid options: 'en', 'es'.

    Returns:
        data (dict): 
            A dictionary containing all the author information in the selected language.
    '''
    go_to_project_root()
    return __import_from__(RAW_DATA_PATH + lang)

def get_processed_data(lang='en'):
    go_to_project_root()
    file_list = [f for f in listdir(PREPROCESSED_DATA_PATH) if isfile(join(PREPROCESSED_DATA_PATH, f)) and f.split(".")[-1] == "json"]

    authors = {}
    for file in file_list:
        f = open(join(PREPROCESSED_DATA_PATH, file), "r")
        lines = "\n".join(f.readlines())
        f.close()
        authors[file.split(".")[0]] = jsonpickle.decode(lines, classes=Author)

    return authors

def exportJSON(author):
    '''
    Exports an author object to a JSON file
    Parameters: 
        author (Author):
            The author to be serialized
    Export: None
    '''
    path = PREPROCESSED_DATA_PATH
    with open(f"{path}{author.author_id}.json", "w") as file:
        file.writelines(__convert_to_JSON__(author))
        file.close()
    return

def __convert_to_JSON__(author):
    return jsonpickle.encode(author)