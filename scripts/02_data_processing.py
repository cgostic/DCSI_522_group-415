# author: Keanna Knebel 
# date: 2020-01-22

'''This script reads in two .csv files with the same columns,
 adds an outcome column, and combines rejected plates with a random
 sample of 2000 observations from accepted plates. The combined dataset
 is split into train, validate, and test sets; followed by
 feature transformtion with CountVectorizer. This script takes
 a file_path, 2 filenames of .csv files for the unprocessed data, 
 and 7 filenames of .csv files for combined and transformed data splits. 

Usage: scripts/02_data_processing.py --file_path=<file_path> --accepted_plates_csv=<accepted_plates_csv> --rejected_plates_csv=<rejected_plates_csv> --reduced_plates_csv=<reduced_plates_csv> --X_test_csv=<X_test_csv> --X_train_csv=<X_train_csv> --X_validate_csv=<X_validate_csv> --y_test_csv=<y_test_csv> --y_train_csv=<y_train_csv> --y_validate_csv=<y_validate_csv> 

Options:
--file_path=<file_path>  Path to data folder of .csv files
--accepted_plates_csv=<accepted_plates_csv> filename of .csv with positive target observations
--rejected_plates_csv=<rejected_plates_csv> filename of .csv with negative target observations
--reduced_plates_csv=<reduced_plates_csv> filename of .csv containing combined and reduced data
--X_test_csv=<X_test_csv> filename of .csv for processed test split features
--X_train_csv=<X_train_csv> filename of .csv for processed train split features
--X_validate_csv=<X_validate_csv> filename of .csv for processed validate split features
--y_test_csv=<y_test_csv> filename of .csv for test split target values
--y_train_csv=<y_train_csv> filename of .csv for train split target values
--y_validate_csv=<y_validate_csv> filename of .csv for validate split target values
'''

import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from docopt import docopt

opt = docopt(__doc__)

def main(file_path, accepted_plates_csv, rejected_plates_csv, reduced_plates_csv, X_test_csv, X_train_csv, X_validate_csv, y_test_csv, y_train_csv, y_validate_csv):
  
  # Read in csv files and add outcome column as either "accepted" or "rejected"
  # for each csv
  # file_path = 'data/'
  # accepted_plates_csv = 'accepted_plates.csv'
  # rejected_plates_csv = 'rejected_plates.csv'
  # reduced_plate_csv = 'full_vanity_plate_data.csv'
  # X_test_csv = 'X_test.csv'
  # X_train_csv = 'X_train.csv'
  # X_validate_csv = 'X_validate.csv'
  # y_test_csv = 'y_test.csv'
  # y_train_csv = 'y_train.csv'
  # y_validate_csv = 'y_validate.csv'
  
  accepted_df =pd.read_csv(file_path + accepted_plates_csv, index_col = 0)
  accepted_df['outcome'] = 'accepted'
  rejected_df = pd.read_csv(file_path + rejected_plates_csv, index_col = 0)
  rejected_df['outcome'] = 'rejected'
  
  # Undersample accepted observations (by taking random sample of 2000)
  reduced_accepted = accepted_df.sample(n = 2000, random_state = 415)
  
  # tests for column addition and df join
  assert reduced_accepted.shape[0] == 2000, "should be 2000 observations"
  assert reduced_accepted['outcome'].all() == 'accepted', "outcome should all be 'accepted'"
  assert rejected_df['outcome'].all() == 'rejected', "outcome should all be 'rejected'"
  
  # combine dataframe and save
  combo_df = reduced_accepted.append(rejected_df)
  combo_df.to_csv(file_path + reduced_plates_csv)
  
  # split data into train and test sets
  X = combo_df['plate']
  y = combo_df['outcome']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 415)
  
  # test for randomization during df split
  assert set(y_train) == {'accepted', 'rejected'}, "training set contains both outcomes"
  
  # split data into train and validate sets
  X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 415)
  
  # export split datasets to csv
  X_train.to_csv(file_path + X_train_csv, header = True)
  X_validate.to_csv(file_path + X_validate_csv, header = True)
  X_test.to_csv(file_path + X_test_csv, header = True)
  y_train.to_csv(file_path + y_train_csv, header = True)
  y_validate.to_csv(file_path + y_validate_csv, header = True)
  y_test.to_csv(file_path + y_test_csv, header = True)
  
  # # export split datasets to csv
  # X_train.to_csv(file_path + 'X_train.csv', header = True)
  # X_validate.to_csv(file_path + 'X_validate.csv', header = True)
  # X_test.to_csv(file_path + 'X_test.csv', header = True)
  # y_train.to_csv(file_path + 'y_train.csv', header = True)
  # y_validate.to_csv(file_path + 'y_validate.csv', header = True)
  # y_test.to_csv(file_path + 'y_test.csv', header = True)

if __name__ == "__main__":
    main(opt["--file_path"], opt["--accepted_plates_csv"], opt["--rejected_plates_csv"], 
    opt["--reduced_plates_csv"], opt["--X_test_csv"], opt["--X_train_csv"], 
    opt["--X_validate_csv"], opt["--y_test_csv"], opt["--y_train_csv"], opt["--y_validate_csv"])
