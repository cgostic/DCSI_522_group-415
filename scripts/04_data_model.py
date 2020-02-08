# author: Furqan Khan
# date: 2020-01-24

'''This script reads six .csv files.


Usage: scripts/04_data_model.py --file_path_read=<file_path_read> --filename_x_train=<filename_x_train> --filename_x_validate=<filename_x_validate> --filename_x_test=<filename_x_test> --filename_y_train=<filename_y_train> --filename_y_validate=<filename_y_validate> --filename_y_test=<filename_y_test> --filename_path_write=<filename_path_write>

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
    # file_path_read = 'data/processed/'
    # filename_x_train = 'X_train.csv'
    # filename_x_validate = 'X_validate.csv'
    # filename_x_test = 'X_test.csv'
    # filename_y_train = 'y_train.csv'
    # filename_y_validate = 'y_validate.csv'
    # filename_y_test = 'y_test.csv'
    # file_path_write = 'results/'
    
    # command line usage: python scripts/04_data_model.py --file_path_read="data/processed/" --filename_x_train="X_train.csv" --filename_x_validate="X_validate.csv" --filename_x_test="X_test.csv" --filename_y_train="y_train.csv" --filename_y_validate="y_validate.csv" --filename_y_test="y_test.csv" --filename_path_write="results/"
    
    # read the training, testing, and validation data
    X_train = np.squeeze(pd.read_csv(file_path_read+filename_x_train, index_col=0))
    X_validate = np.squeeze(pd.read_csv(
        file_path_read+filename_x_validate, index_col=0))
    X_test = np.squeeze(pd.read_csv(file_path_read+filename_x_test, index_col=0))

    y_train = np.squeeze(pd.read_csv(file_path_read+filename_y_train, index_col=0))
    y_validate = np.squeeze(pd.read_csv(
        file_path_read+filename_y_validate, index_col=0))
    y_test = np.squeeze(pd.read_csv(file_path_read+filename_y_test, index_col=0))
    

    # Test that data read in is correct to reproduce results
    assert X_train.shape == (2332,), "X_train is incorrect shape"
    assert sum(X_train.index) == 87392513, "X_train has incorrect observations"
    assert y_train.shape == (2332,), "y_train is incorrect shape"
    assert X_validate.shape == (584,), "X_validate has incorrect shape"
    assert sum(X_validate.index) == 21381421, "X_validate has correct observations"
    assert y_validate.shape == (584,),"y_validate has incorrect shape"


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
        color = alt.Color('variable:N', legend = alt.Legend(title = ""))
        )

    point = alt.Chart(tr_v_plot_df).mark_point().encode(
        x = alt.X('Number of Features:Q'),
        y = alt.Y('value:Q', title = 'Accuracy score'),
        color = alt.Color('variable:N', legend = alt.Legend(title = ""))
        )

    text = alt.Chart(tr_v_plot_df.query('variable == "Cross-Val Training error (cv = 10)"')
    ).mark_text(dy = 13
    ).encode(
        x = alt.X('Number of Features:Q'),
        y = alt.Y('value:Q', title = 'Accuracy score'),
        text = alt.Text('n-gram_range:N'))

    (line + point + text).configure_axis(labelFontSize=15,titleFontSize=15
        ).configure_header(labelFontSize=15
        ).configure_title(fontSize=20, anchor = 'middle'
        ).configure_legend(
            orient = 'none',
            fillColor = 'white', 
            legendX = 475,
            legendY = 250
        ).properties(width = 700,
         background = 'white', title = 'CV Training and Validation Error by Number of Features'
         ).save(file_path_write + 'train_val_error.png', scale_factor = 2)

    # Train model with chosen n-gram length range (2,2)
    cv_mnb = CountVectorizer(analyzer = 'char', ngram_range = (2,2))
    X_train_t = cv_mnb.fit_transform(X_train)
    X_val_t = cv_mnb.transform(X_validate)
    X_test_t = cv_mnb.transform(X_test)

    # Report VALIDATION error for model with optimal n-gram length range
    mnb = MultinomialNB().fit(X_train_t, y_train)

    clf_val_report = classification_report(
    y_validate, mnb.predict(X_val_t), output_dict=True)
    clf_val_report_df = pd.DataFrame(clf_val_report).transpose().round(3)
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
    clf_test_report_df = pd.DataFrame(clf_test_report).transpose().round(3)
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

    ng_l_best = [(2,2)]
    
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
        # Find most positive weight (where 'accepted' is the negative class, and 'rejected' is positive)
        max_coef = max(feat_df['weights'])
        positive = feat_df.sort_values(by = 'weights', ascending = False)['features'].head(50).reset_index(drop = True)
        features_50 = pd.concat([positive.iloc[:10].reset_index(drop = True),
            positive.iloc[10:20].reset_index(drop = True),
            positive.iloc[20:30].reset_index(drop = True),
            positive.iloc[30:40].reset_index(drop = True),
            positive.iloc[40:50].reset_index(drop = True)], axis = 1)
        file_name_html = file_path_write + 'predictors_'+str(ng)[1]+'_'+str(ng)[-2]+'.html'
        file_name_png = file_path_write + 'predictors_'+str(ng)[1]+'_'+str(ng)[-2]+'.png'
        features_50.to_html(file_name_html, index = False, header = False)
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
