# author: Cari Gostic
# date: 2020-01-16

'''This script downloads 2 .csv files with the same columns and, 
combines them into a single dataframe, and exports this dataframe 
as a .csv into the \data folder. This script takes the filename, 
2 urls, and a local filepath as arguments.

Usage: scripts/download_data.py --file_path=<file_path> --url1=<url1> --url2=<url2>

Options:
--file_path=<file_path>  Path (including filename) to the csv file.
--url1=<url1>            URL of first csv
--url2=<url2>            URL of second csv
'''

# URLS to pass in for DSCI 522
# https://raw.githubusercontent.com/datanews/license-plates/master/accepted-plates.csv
# https://raw.githubusercontent.com/datanews/license-plates/master/rejected-plates.csv

import pandas as pd
import numpy as np
from docopt import docopt

opt = docopt(__doc__)

def main(file_path, url1, url2):
    """
    Combines two .csv files from url1 and url2
    into single dataframe and writes a .csv 
    to the provided filepaths. .csv's to be
    combined need to have the same columns.

    Parameters
    ----------
    file_path : str
        The local filepath (including file name)
    
    url1 : str
        URL to a .csv file
    
    url2 : str
        URL to a .csv file

    Returns
    -------

    Examples
    --------
    main('data/alphabet_data.csv', 
        'https://public-data.com/url_abc.csv',
        'https://public-data.com/url_def.csv')
    """
    df_1=pd.read_csv(url1)
    df_2 = pd.read_csv(url2)

    df_combo = df_1.append(df_2)
    df_combo.to_csv(file_path)

if __name__ == "__main__":
    main(opt["--file_path"], opt["--url1"], opt["--url2"])


