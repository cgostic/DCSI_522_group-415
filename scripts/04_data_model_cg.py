# author: Furqan Khan
# date: 2020-01-24

'''This script reads six .csv files.


Usage: scripts/data_model.py --file_path_read=<file_path_read> --filename_x_train=<filename_x_train> --filename_x_validate=<filename_x_validate> --filename_x_test=<filename_x_test> --filename_y_train=<filename_y_train> --filename_y_validate=<filename_y_validate> --filename_y_test=<filename_y_test> --filename_path_write=<filename_path_write>

Options:
--file_path_read=<file_path_read>  Path to data folder of .csv files
--filename_x_train=<filename_x_train> filename of .csv with x_train data
--filename_x_validate=<filename_x_validate> filename of .csv with x_validate data
--filename_x_test=<filename_x_test> filename of .csv with x_test data
--filename_y_train=<filename_y_train> filename of .csv with y_train data
--filename_y_validate=<filename_y_validate> filename of .csv with y_validate data
--filename_y_test=<filename_y_test> filename of .csv with y_test data
--filename_path_write=<filename_path_write> path to write image files
'''

import pandas as pd 
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt 
import subprocess # pip install subprocess.run
from docopt import docopt
import altair as alt

opt = docopt(__doc__)

def main(file_path_read, filename_x_train, filename_x_validate, filename_x_test, filename_y_train, filename_y_validate, filename_y_test, file_path_write):
  
    # Read csv files of X and y components of data
    # file_path_read = 'data/'
    # filename_x_train = 'X_train.csv'
    # filename_x_validate = 'X_validate.csv'
    # filename_x_test = 'X_test.csv'
    # filename_y_train = 'y_train.csv'
    # filename_y_validate = 'y_validate.csv'
    # filename_y_test = 'y_test.csv'
    # file_path_write = 'docs/imgs/'
    
    # command line usage: python scripts/04_data_model_cg.py --file_path_read="data/" --filename_x_train="X_train.csv" --filename_x_validate="X_validate.csv" --filename_x_test="X_test.csv" --filename_y_train="y_train.csv" --filename_y_validate="y_validate.csv" --filename_y_test="y_test.csv" --filename_path_write="docs/imgs/"
    
    # read the training, testing, and validation data
    X_train = np.squeeze(pd.read_csv(file_path_read+filename_x_train, index_col=0))
    X_validate = np.squeeze(pd.read_csv(
        file_path_read+filename_x_validate, index_col=0))
    X_test = np.squeeze(pd.read_csv(file_path_read+filename_x_test, index_col=0))

    y_train = np.squeeze(pd.read_csv(file_path_read+filename_y_train, index_col=0))
    y_validate = np.squeeze(pd.read_csv(
        file_path_read+filename_y_validate, index_col=0))
    y_test = np.squeeze(pd.read_csv(file_path_read+filename_y_test, index_col=0))


    pipeline = Pipeline(steps=[
    ('vect', CountVectorizer(analyzer = 'char')),
    ('mnb', MultinomialNB())])

    param_grid = {'vect__ngram_range': ((2, 2), (2,3), (3, 3), (2,4), (3,4), (4, 4), (4,5), (5, 5), (6, 6), (7, 7),
                                    (8, 8))}

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    cv = CountVectorizer(analyzer = 'char', ngram_range=grid_search.best_params_['vect__ngram_range'])
    
    #Report best param chosen through gridSearch
    grid_search.best_params_['vect__ngram_range']

    # Check triaining and validation error for various
    # n-gram length ranges to evaluate GridSearch's
    # pick
    ng_list = [(2, 2), (2,3), (3, 3), (2,4), (3,4), (4, 4), 
            (3,5), (4,5), (2,5), (5, 5), (2,6), (3,6), 
            (4,6), (5,6), (6, 6), (2,7), (3,7), (4,7), 
            (5,7), (6,7), (7, 7), (8, 8)]
    num_f = []
    tr_err = []
    v_err = []
    ng_l = []
    for ng in ng_list:
        cv = CountVectorizer(analyzer = 'char', ngram_range=ng)
        X_train_trs = cv.fit_transform(X_train)
        X_validate_trs = cv.transform(X_validate)
        num_f.append(len(cv.get_feature_names()))
        mnb = MultinomialNB().fit(X_train_trs, y_train)
        tr_err.append(np.mean(cross_val_score(mnb, X_train_trs, y_train, cv = 10)))
        v_err.append(mnb.score(X_validate_trs, y_validate))
        ng_l.append(str(ng))

    tr_v_df = pd.DataFrame({'Number of Features':num_f, 'Cross-Val Training error (cv = 10)':tr_err, 'Validation error':v_err, 'n-gram_range':ng_l})
    tr_v_plot_df = tr_v_df.melt(['Number of Features', 'n-gram_range'])

    # Plot training and validation error vs. number of features
    line = alt.Chart(tr_v_plot_df).mark_line().encode(
        x = alt.X('Number of Features:Q'),
        y = alt.Y('value:Q', title = 'Accuracy score'),
        color = 'variable:N'
        )

    point = alt.Chart(tr_v_plot_df).mark_point().encode(
        x = alt.X('Number of Features:Q'),
        y = alt.Y('value:Q', title = 'Accuracy score'),
        color = 'variable:N'
        )

    text = alt.Chart(tr_v_plot_df.query('variable == "Cross-Val Training error (cv = 10)"')).mark_text(dy = 13).encode(
        x = alt.X('Number of Features:Q'),
        y = alt.Y('value:Q', title = 'Accuracy score'),
        text = alt.Text('n-gram_range:N'))

    (line + point + text).properties(width = 700,
         background = 'white').save(file_path_write + 'train_val_error.png', scale_factor = 2)

    # Train model with chosen n-gram length range (2,2)
    cv_mnb = CountVectorizer(analyzer = 'char', ngram_range = (2,2))
    X_train_t = cv_mnb.fit_transform(X_train)
    X_val_t = cv_mnb.transform(X_validate)
    X_test_t = cv_mnb.transform(X_test)

    # Report VALIDATION error for model with optimal n-gram length range
    mnb = MultinomialNB().fit(X_train_t, y_train)

    clf_val_report = classification_report(
    y_validate, mnb.predict(X_val_t), output_dict=True)
    clf_val_report_df = pd.DataFrame(clf_val_report).transpose()
    clf_val_report_df = clf_val_report_df.iloc[:3, :-1]

    # export cl_val_report_df and predictor_df as image file
    file_name_html = file_path_write + 'clf_val_report.html'
    file_name_png = file_path_write + 'clf_val_report.png'
    clf_val_report_df.to_html(file_name_html)
    subprocess.call(
        ['wkhtmltoimage -f png --width 0 ' + file_name_html + " " + file_name_png], shell=True)

    # Report TESTING error for model with optimal n-gram length range
    # Test scores
    clf_test_report = classification_report(
    y_test, mnb.predict(X_test_t), output_dict=True)
    clf_test_report_df = pd.DataFrame(clf_test_report).transpose()
    clf_test_report_df = clf_test_report_df.iloc[:3, :-1]
    clf_test_report_df

    # export cl_test_report_df and predictor_df as image file
    file_name_html = file_path_write + 'clf_test_report.html'
    file_name_png = file_path_write + 'clf_test_report.png'
    clf_test_report_df.to_html(file_name_html)
    subprocess.call(
    ['wkhtmltoimage -f png --width 0 ' + file_name_html + " " + file_name_png], shell=True)

    # Export tables with strongest predictors for various n-gram ranges
    ng_l_best = [(2,2), (3,3), (4,4)]

    for ng in ng_l_best:
        cv_mnb = CountVectorizer(analyzer = 'char', ngram_range = ng)
        X_train_t = cv_mnb.fit_transform(X_train)
        X_val_t = cv_mnb.transform(X_validate)
        X_test_t = cv_mnb.transform(X_test)
        mnb = MultinomialNB().fit(X_train_t, y_train)
        # join feature names with respective weights into a dataframe
        vocab = cv_mnb.get_feature_names()
        weights = mnb.coef_.flatten()
        feat_df = pd.DataFrame({'features':vocab, 'weights':weights})
        # Find most negative weight
        least_coef = min(feat_df['weights'])
        # check how many features share that weight
        print("Number of features tied for strongest predictor:", len(feat_df.query('weights == '+str(least_coef))))
        negative = feat_df.query('weights == '+str(least_coef))[['features']].sample(n = 50, random_state = 415).reset_index(drop=True)
        #negative = negative.sample(n = 50).reset_index(drop=True)
        features_100 = pd.concat([negative.iloc[:10].reset_index(drop = True),
            negative.iloc[10:20].reset_index(drop = True),
            negative.iloc[20:30].reset_index(drop = True),
            negative.iloc[30:40].reset_index(drop = True),
            negative.iloc[40:50].reset_index(drop = True)], axis = 1)
        file_name_html = file_path_write + 'predictors_'+str(ng)[1]+'_'+str(ng)[-2]+'.html'
        file_name_png = file_path_write + 'predictors_'+str(ng)[1]+'_'+str(ng)[-2]+'.png'
        features_100.to_html(file_name_html, index = False, header = False)
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
    opt["--filename_path_write"])
