## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
from utils import go_to_project_root

RAW_DATA_PATH = "data/raw/"

class Author:
    def __init__(self, author_id, tweets, truth):
        self.author_id = author_id
        self.truth = truth
        self.tweets = tweets

def __get_truth_vals__(directory):
    truth_dict = {}
    for f in listdir(directory):
        if f.split(".")[-1] != "xml":
            truth_file = open(join(directory, f), "r")
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
    tweets_file = open(filepath, "r")
    content = tweets_file.read()
    soup = BeautifulSoup(content, 'xml')
    tweets = soup.find_all('document')
    return [t.get_text() for t in tweets]

def get_raw_data(lang):
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