from os import getcwd, chdir
from re import split

def go_to_project_root():
    while(split(r"[\/\\]", getcwd())[-1] != "NLP2"):
        chdir('../')