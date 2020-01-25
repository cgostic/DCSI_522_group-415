# author: Furqan Khan
# date: 2020-01-24

'''This script reads six .csv files.


Usage: scripts/data_model.py --file_path_read=<file_path_read> --filename_x_train=<filename_x_train> --filename_x_validate=<filename_x_validate> --filename_x_test=<filename_x_test> --filename_y_train=<filename_y_train> --filename_y_validate=<filename_y_validate> --filename_y_test=<filename_y_test> --filename_path_write=<filename_path_write> --filename_cl_table=<filename_cl_table> --filename_predictor_table=<filename_predictor_table>

Options:
--file_path_read=<file_path_read>  Path to data folder of .csv files
--filename_x_train=<filename_x_train> filename of .csv with x_train data
--filename_x_validate=<filename_x_validate> filename of .csv with x_validate data
--filename_x_test=<filename_x_test> filename of .csv with x_test data
--filename_y_train=<filename_y_train> filename of .csv with y_train data
--filename_y_validate=<filename_y_validate> filename of .csv with y_validate data
--filename_y_test=<filename_y_test> filename of .csv with y_test data
--filename_path_write=<filename_path_write> path to write image files
--filename_cl_table=<filename_cl_table> filename of classification report
--filename_predictor_table=<filename_predictor_table> filename of top predictors for plate acceptance or rejection
'''

import pandas as pd 
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import subprocess # pip install subprocess.run
from docopt import docopt

opt = docopt(__doc__)

def main(file_path_read, filename_x_train, filename_x_validate, filename_x_test, filename_y_train, filename_y_validate, filename_y_test, file_path_write, filename_cl_table, filename_predictor_table):
  
  # Read csv files of X and y components of data
  # file_path_read = 'data/'
  # filename_x_train = 'X_train.csv'
  # filename_x_validate = 'X_validate.csv'
  # filename_x_test = 'X_test.csv'
  # filename_y_train = 'y_train.csv'
  # filename_y_validate = 'y_validate.csv'
  # filename_y_test = 'y_test.csv'
  # file_path_write = 'docs/imgs/'
  # filename_cl_table = 'classification_report'
  # filename_predictor_table = 'best_predictors'
  
  # command line usage: python scripts/04_data_model.py --file_path_read="data/" --filename_x_train="X_train.csv" --filename_x_validate="X_validate.csv" --filename_x_test="X_test.csv" --filename_y_train="y_train.csv" --filename_y_validate="y_validate.csv" --filename_y_test="y_test.csv" --filename_path_write="docs/imgs/" --filename_cl_table="classification_report" --filename_predictor_table="best_predictors"
  
  # read the training, testing, and validation data
  X_train = np.squeeze(pd.read_csv(file_path_read+filename_x_train, index_col=0))
  X_validate = np.squeeze(pd.read_csv(
      file_path_read+filename_x_validate, index_col=0))
  X_test = np.squeeze(pd.read_csv(file_path_read+filename_x_test, index_col=0))
  X_trainvalidate = pd.concat([X_train, X_validate])
  y_train = np.squeeze(pd.read_csv(file_path_read+filename_y_train, index_col=0))
  y_validate = np.squeeze(pd.read_csv(
      file_path_read+filename_y_validate, index_col=0))
  y_test = np.squeeze(pd.read_csv(file_path_read+filename_y_test, index_col=0))
  y_trainvalidate = pd.concat([y_train, y_validate])

  # optiimize hyperparameters for CountVectorize
  pipeline = Pipeline(steps=[
      ('vect', CountVectorizer()),
      ('mnb', MultinomialNB())])

  param_grid = {'vect__analyzer': ('char_wb', 'char'),
                'vect__ngram_range': ((2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
                                      (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (7, 8),
                                      (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7),
                                      (5, 8), (6, 7), (6, 8) )}
  grid_search = GridSearchCV(pipeline, param_grid, cv=5)
  grid_search.fit(X_train, y_train).best_params_['vect__analyzer']

  # transform trainvalidate data with CountVectorizer
  countv = CountVectorizer(analyzer=grid_search.best_params_[
                          'vect__analyzer'], ngram_range=grid_search.best_params_['vect__ngram_range'])
  X_trainvalidate_trs = countv.fit_transform(X_trainvalidate)
  X_test_trs = countv.transform(X_test)

  # fit model with trainvalidate dataset
  mnb = MultinomialNB()
  mnb.fit(X_trainvalidate_trs, y_trainvalidate)
  print('Training accuracy is: ', mnb.score(X_trainvalidate_trs, y_trainvalidate))
  print('Test accuracy is: ', mnb.score(X_test_trs, y_test))
  clf_report = classification_report(
      y_test, mnb.predict(X_test_trs), output_dict=True)
  clf_report_df = pd.DataFrame(clf_report).transpose()

  # Get the top 25 predictors for plate rejection and acceptance 
  vocab = countv.get_feature_names()
  weights = mnb.coef_.flatten()

  inds = np.argsort(mnb.coef_.flatten())
    
  negative = [vocab[index] for index in inds[:25]]
  positive = [vocab[index] for index in inds[-25:]]

  predictor_df = pd.DataFrame({'Rejection features':negative, 'Acceptance features':positive})

  # export cl_report_df and predictor_df as image file
  file_name_html = file_path_write + filename_cl_table + '.html'
  file_name_png = file_path_write + filename_cl_table + '.png'
  clf_report_df.to_html(file_name_html)
  subprocess.call(
      ['wkhtmltoimage -f png --width 0 ' + file_name_html + " " + file_name_png], shell=True)

  file_name_html = file_path_write + filename_predictor_table + '.html'
  file_name_png = file_path_write + filename_predictor_table + '.png'
  predictor_df.to_html(file_name_html)
  subprocess.call(
      ['wkhtmltoimage -f png --width 0 ' + file_name_html + " " + file_name_png], shell=True)

  
if __name__ == "__main__":
    main(opt["--file_path_read"],
    opt["--filename_x_train"],
    opt["--filename_x_validate"],
    opt["--filename_x_test"],
    opt["--filename_y_train"],
    opt["--filename_y_validate"],
    opt["--filename_y_test"],
    opt["--filename_path_write"],
    opt["--filename_cl_table"],
    opt["--filename_predictor_table"])
