## Natural Language Processing 2 Final
## Sara, Flora, Giorgos

import data
import numpy as np
import pandas as pd

filepath = '/data/processed/'

def convert_to_df(authors):
    '''
    Converts the given authors to a pandas DataFrame.
    Parameters:
        authors (Author dict):
            The authors to be converted.
    Returns:
        pandas DataFrame
    '''
    table = np.zeros((len(list(authors)), 20))
    df = pd.DataFrame(table)

    return df
