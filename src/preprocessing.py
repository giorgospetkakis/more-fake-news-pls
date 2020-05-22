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
    table = np.zeros((len(list(authors.values())), 6))

    for i, a in enumerate(list(authors.values())):
        table[i] = [a.max_similar, a.min_similar, a.mean_similar, a.number_identical, a.most_common_ner_score, a.truth]

    df = pd.DataFrame(table, columns=["max_similar", "min_similar", "mean_similar", "number_identical", "most_common_ner_score", "truth"])

    # Enable to export to file
    if export:
        go_to_project_root()
        df.to_csv(filepath+"processed.csv")

    return df
