'''This script reads in two .csv files with the same columns,
 adds an outcome column, and combines rejected plates with accepted
 plates. The combined dataset is split into train, validate, and test sets; 
 Then a the accepted class is undersampled in the training dataset.
  This script takes a file_path for the .csv data files, 2 filenames for the 
 unprocessed data and a filename for the reduced dataset.

Usage: scripts/02_data_processing.py --file_path_read=<file_path_read> --file_path_write=<file_path_write>  --accepted_plates_csv=<accepted_plates_csv> --rejected_plates_csv=<rejected_plates_csv> 

Options:
--file_path_read=<file_path_write>  Path to raw data folder of .csv files
--file_path_write=<file_path_write> Path to processed data folder
--accepted_plates_csv=<accepted_plates_csv> filename of .csv with positive target observations
--rejected_plates_csv=<rejected_plates_csv> filename of .csv with negative target observations
'''

import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from docopt import docopt

opt = docopt(__doc__)

def main(file_path_read, file_path_write, accepted_plates_csv, rejected_plates_csv):

  # Read in csv files and add outcome column as either "accepted" or "rejected"
  # for each csv
  # file_path_read = 'data/raw/'
  # file_path_write = 'data/processed/'
  # accepted_plates_csv = 'accepted_plates.csv'
  # rejected_plates_csv = 'rejected_plates.csv'

  # file_path_read = "data/raw/"
  # accepted_plates_csv = "accepted_plates.csv"
  # rejected_plates_csv = "rejected_plates.csv"
  # Read in csv files and add outcome column as either "accepted" or "rejected"
  accepted_df =pd.read_csv(file_path_read + accepted_plates_csv, index_col = 0)
  accepted_df['outcome'] = 'accepted'
  rejected_df = pd.read_csv(file_path_read + rejected_plates_csv, index_col = 0)
  rejected_df['outcome'] = 'rejected'

  # tests for column addition and df join
  assert accepted_df.shape[0] == 131990, "should be 131990 observations"
  assert accepted_df['outcome'].all() == 'accepted', "outcome should all be 'accepted'"
  assert rejected_df['outcome'].all() == 'rejected', "outcome should all be 'rejected'"

  # combine dataframe and save
  combo_df = accepted_df.append(rejected_df)
  #combo_df.to_csv(file_path_write + reduced_plates_csv)

  # split data into train and test sets. Use stratify to ensure each set has the "rejected" class.
  X = combo_df['plate']
  y = combo_df['outcome']
  X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, 
                                                                test_size = 0.2, 
                                                                random_state = 415, 
                                                                stratify = y)

  # split data into train and validate sets. Use stratify to ensure each set has the "rejected" class.
  X_train_full, X_validate, y_train_full, y_validate = train_test_split(X_train_full, 
                                                                        y_train_full, 
                                                                        test_size = 0.2, 
                                                                        random_state = 415, 
                                                                        stratify = y_train_full)

  # test for randomization during df split
  # assert set(y_train_full) == {'accepted', 'rejected'}, "training set contains both outcomes"

  # Undersample accepted observations in training set(by taking random sample of 1200)
  X_y_train = pd.concat([X_train_full, y_train_full], axis = 1)
  # Split into accepted vs. rejected datasets
  X_y_train_a = X_y_train.query('outcome == "accepted"').copy()
  X_y_train_r = X_y_train.query('outcome == "rejected"').copy()
  # Undersample 1200 examples of accepted class
  reduced_accepted = X_y_train_a.sample(n = 1200, random_state = 415)
  # Append rejected examples to reduced set of accepted examples
  X_y_train_reduced = reduced_accepted.append(X_y_train_r)
  # Split bak out into X_train and y_train
  X_train = X_y_train_reduced['plate']
  y_train = X_y_train_reduced['outcome']
  
  # export split datasets to csv
  X_train.to_csv(file_path_write + 'X_train.csv', header = True)
  X_validate.to_csv(file_path_write + 'X_validate.csv', header = True)
  X_test.to_csv(file_path_write + 'X_test.csv', header = True)
  y_train.to_csv(file_path_write + 'y_train.csv', header = True)
  y_validate.to_csv(file_path_write + 'y_validate.csv', header = True)
  y_test.to_csv(file_path_write + 'y_test.csv', header = True)

  # print('X_train shape: ', X_train.shape)
  # print('sum X_train ind: ', sum(X_train.index))
  # print('y_train shape: ', X_train.shape)
  # print('X_val shape: ', X_validate.shape)
  # print('sum X_val ind: ', sum(X_validate.index))

if __name__ == "__main__":
    main(opt["--file_path_read"], opt["--file_path_write"], opt["--accepted_plates_csv"], opt["--rejected_plates_csv"])