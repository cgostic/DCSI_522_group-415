# author: Keanna Knebel 
# date: 2020-01-22

'''This script reads in two .csv files with the same columns,
 adds an outcome column, and combines file 2 with a random
 sample of 2000 observations from file 1. The combined dataset
 is split into train, validate, and test sets; followed by
 feature transformtion with CountVectorizer. This script takes
 the a file_path and 3 filenames of .csv files for the unprocessed
 and processed data.


Usage: scripts/data_processing.py --file_path=<file_path> --filename_1=<filename_1> --filename_2=<filename_2> --filename_3=<filename_3>

Options:
--file_path=<file_path>  Path to data folder of .csv files
--filename_1=<filename_1> filename of .csv with positive target observations
--filename_2=<filename_2> filename of .csv with negative target observations
--filename_3=<filename_3> filename of .csv containing processed data
'''

import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from docopt import docopt

opt = docopt(__doc__)

def main(file_path, filename_1, filename_2, filename_3):
  
  # Read in csv files and add outcome column as either "accepted" or "rejected"
  # for each csv
  # file_path = 'data/'
  # filename_1 = 'accepted_plates.csv'
  # filename_2 = 'rejected_plates.csv'
  # filename_3 = 'full_vanity_plate_data.csv'
  
  accepted_df =pd.read_csv(file_path + filename_1, index_col = 0)
  accepted_df['outcome'] = 'accepted'
  rejected_df = pd.read_csv(file_path + filename_2, index_col = 0)
  rejected_df['outcome'] = 'rejected'
  
  # Undersample accepted observations (by taking random sample of 2000)
  reduced_accepted = accepted_df.sample(n = 2000, random_state = 415)
  
  # tests for column addition and df join
  assert reduced_accepted.shape[0] == 2000, "should be 2000 observations"
  assert reduced_accepted['outcome'].all() == 'accepted', "outcome should all be 'accepted'"
  assert rejected_df['outcome'].all() == 'rejected', "outcome should all be 'rejected'"
  
  # combine dataframe and save
  combo_df = reduced_accepted.append(rejected_df)
  combo_df.to_csv(file_path + filename_3)
  
  # split data into train and test sets
  X = combo_df['plate']
  y = combo_df['outcome']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 415)
  
  # test for randomization during df split
  assert set(y_train) == {'accepted', 'rejected'}, "training set contains both outcomes"
  
  # split data into train and validate sets
  X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 415)
  
  # feature engineering with count vectorizer
  cv = CountVectorizer(analyzer='char', ngram_range=(2,8))
  X_train_transformed = cv.fit_transform(X_train)
  X_validate_transformed = cv.transform(X_validate)
  X_test_transformed = cv.transform(X_test)

if __name__ == "__main__":
    main(opt["--file_path"], opt["--filename_1"], opt["--filename_2"], opt["--filename_3"])
