## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
from util import go_to_project_root

class Author:
    def __init__(self, author_id, tweets, truth):
        self.author_id = author_id
        self.truth = truth
        self.tweets = tweets

def get_truth_vals(directory):
    truth_dict = {}
    for f in listdir(directory):
        if f.split(".")[-1] != "xml":
            truth_file = open(join(directory, f), "r")
            for line in truth_file.readlines():
                split = line.split(":::")
                truth_dict[split[0]] = int(split[1].strip())
            break
    return truth_dict

def import_from(directory):
    file_list = [f for f in listdir(directory) if isfile(join(directory, f)) and f.split(".")[-1] == "xml"]
    truth_vals = get_truth_vals(directory)
    author_dict = {}
    for file in file_list:
        author_id = file.split(".")[0]
        author_dict[author_id] = Author(author_id, parse_tweets(join(directory, file)), truth_vals[author_id])
    return author_dict

def parse_tweets(filepath):
    tweets_file = open(filepath, "r")
    content = tweets_file.read()
    soup = BeautifulSoup(content, 'xml')
    tweets = soup.find_all('document')
    
    return [t.get_text() for t in tweets]

RAW_DATA_EN = "data/raw/en"
RAW_DATA_ES = "data/raw/es"

go_to_project_root()
data_en = import_from(RAW_DATA_EN)
data_es = import_from(RAW_DATA_ES)