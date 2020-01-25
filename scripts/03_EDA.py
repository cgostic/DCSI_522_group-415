
# author: Cari Gostic, Furqan Khan 
# date: 2020-01-22

'''This script reads in 5 .csv files located in the <file_path_data> folder:
          1. All accepted vanity plates
          2. All rejected vanity plates
          3. Combined rejected and undersampled rejected plates
          4. Feature training data
          5. Target training data
  And produces 3 plots that best add to the discussion on our
  exploratory data analysis. Features are engineered from the 
  training feature set using sci-kit learn's CountVectorizer.
  .png images of created plots are exported to the <file_path_img> 
  folder. 


Usage: scripts/EDA.py --file_path_data=<file_path_data> --accepted_plates_csv=<accepted_plates_csv> --rejected_plates_csv=<rejected_plates_csv> --reduced_plate_csv=<reduced_plate_csv> --X_train_csv=<X_train_csv> --y_train_csv=<y_train_csv> --file_path_img=<file_path_img>

Options:
--file_path_data=<file_path_data>  Path to data folder of .csv files
--accepted_plates_csv=<accepted_plates_csv> filename of .csv with all accepted plates
--rejected_plates_csv=<rejected_plates_csv> filename of .csv with all negative plates
--reduced_plate_csv=<reduced_plate_csv> filename of .csv of undersampled accepted plates combined with rejected plates
--X_train_csv=<X_train_csv> filename of .csv with training feature dataset
--y_train_csv=<y_train_csv> filename of .csv with training target dataset
--file_path_img=<file_path_img> filepath to folder where images should be stored
'''

# file_path_data = 'data/'
# accepted_plates_csv = 'accepted_plates.csv'
# rejected_plates_csv = 'rejected_plates.csv'
# reduced_plate_csv = 'full_vanity_plate_data.csv'
# X_train_csv = 'X_train.csv'
# y_train_csv = 'y_train.csv'
# file_path_img = 'docs/imgs/'

# #### Exploratory data analysis (EDA)
# 
# In this section we perform EDA of the given dataset to use it to answer the research question. 

import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import altair as alt
from docopt import docopt

opt = docopt(__doc__)

def main(file_path_data, accepted_plates_csv, rejected_plates_csv, reduced_plate_csv, X_train_csv, y_train_csv, file_path_img):
    # Load datasets
    full_rejected = pd.read_csv(file_path_data + rejected_plates_csv)
    full_accepted = pd.read_csv(file_path_data + accepted_plates_csv)
    reduced_plate_df=pd.read_csv(file_path_data + reduced_plate_csv)
    X_train = pd.read_csv(file_path_data + X_train_csv, usecols = ['plate'])
    y_train = pd.read_csv(file_path_data + y_train_csv, usecols = ['outcome'])

    # Transform X_train dataset using countvectorizer
    cv = CountVectorizer(analyzer = 'char', ngram_range=(2,8))
    # Make sure data read in as dataframe
    assert type(X_train['plate']) == pd.core.series.Series, 'Need to pass Series into CountVectorizer'

    # fit and transform X_train via CountVectorizer
    X_train_transformed = cv.fit_transform(X_train['plate'])
    # Make sure transformed data is correct shape
    assert X_train_transformed.shape == (2332, 25889), 'sparse matrix should be of shape 2332 x 25889'

    # Explore full dataset of all rejected and accepted plates
    # Create outcome column
    full_rejected['outcome'] = 'rejected'
    full_accepted['outcome'] = 'accepted'

    # Combine accepted and rejected into single df
    plate_df = full_accepted.append(full_rejected)[['date','plate','outcome']]

    # Plot counts of raw data across classes
    examples_per_classification = (alt.Chart(plate_df).mark_bar().encode(
        alt.X("outcome:N", bin=True, axis=alt.Axis(ticks = False, labels = False, title = "")),
        alt.Y("count()", scale=alt.Scale(type='log', base=10), title = 'Count'), 
        column = alt.Column('outcome:N', title = "")
    ).configure_axis(labelFontSize=15,titleFontSize=20
    ).configure_header(labelFontSize=14
    ).configure_title(fontSize=20
    ).properties(title="Number of examples per classification", 
                width=100, 
                height=300,
                background = 'white'))

    examples_per_classification.save(file_path_img+'examples_per_classification.png', scale_factor = 2.0)

    # Plot frequencies of n-gram lengths
    counts = pd.DataFrame({'ngrams':np.array(cv.get_feature_names()), 
        'counts': np.squeeze(np.asarray(X_train_transformed.sum(axis = 0)))})
    # Add column with length of n-gram
    counts['ng_length'] = counts['ngrams'].str.len()

    n_g_len_chart = (alt.Chart(counts.query('ng_length%2 == 0')).mark_bar().encode(
            x = alt.X('counts:O', 
                title = "Frequency of appearance in plates"),
                #scale=alt.Scale(domain = (0,89))),
            y = alt.Y("count()", scale=alt.Scale(type='log', base=10), title = 'Count'),
            facet = alt.Facet('ng_length:N', title = 'n-gram length')
        ).configure_axis(labelFontSize=15,titleFontSize=20
        ).configure_header(labelFontSize=14
        ).configure_title(fontSize=20, anchor = 'middle'
        ).properties(title = "Counts of n-grams by length", 
                    width = 400, 
                    height = 70, 
                    columns = 1,
                    background = 'white'))

    n_g_len_chart.save(file_path_img+'ngram_length_counts.png', scale_factor = 2.0)


    # Find proportion of each class (accepted or rejected) by n-gram length
    counts_by_length = counts.groupby('ng_length').head(200)
    counts_by_length_ngs = list(counts_by_length['ngrams'])
    counts_bl_index = np.array([cv.get_feature_names().index(ng) for ng in counts_by_length_ngs])
    cbl_df = pd.DataFrame(X_train_transformed[:,counts_bl_index].todense(), 
                columns = [cv.get_feature_names()[ng] for ng in counts_bl_index])
    cbl_df['outcome'] = y_train

    # Create dataframe for plotting with ngram, outcome, proportion per outcome, and length
    ng = []
    class_1 = []
    p = []
    ng_length = []
    for col in cbl_df.columns[:-1]:
        num_per_class = cbl_df[cbl_df[col] != 0][[col, 'outcome']].groupby('outcome').count()
        # Add ngram 2x since we're adding both classifications and their
        # respectiv proportions
        ng.extend([col, col])
        # Add classifications
        class_1.extend(['accepted','rejected'])
        total_counts = sum(num_per_class[col])
        # Add feature length 2x
        ng_length.extend([len(col), len(col)])
        # Account for features that only appear in 1 class
        if len(num_per_class.index) == 1:
            if num_per_class.index[0] == 'rejected':
                p.extend([0, num_per_class.at['rejected', col]/total_counts])
            if num_per_class.index[0] == 'accepted':
                p.extend([num_per_class.at['accepted', col]/total_counts, 0])
        else:
            p.extend([num_per_class.at['accepted', col]/total_counts, 
            num_per_class.at['rejected', col]/total_counts])       
    cbl_prop_df = pd.DataFrame({'n-gram': ng, 'class':class_1, 'p':p, 'ng_length':ng_length})

    # Plot distribution of proportion per class per ngram length
    cl = []
    l = [str(i) for i in range(2,9)]
    for i in l:
        cl.append(p_chart_grid(cbl_prop_df.query('ng_length == '+i), i))
    chart_grid = (((cl[0] | cl [1] | cl[2]) & (cl[3] | cl [4] | cl[5]) & (cl[6])
        ).properties(title = "Distribution of Class Proportion by n-gram Length: 200 Samples"
        ).configure_axis(labelFontSize=15,
                        titleFontSize=20
        ).configure_title(fontSize=20, 
                        anchor = 'middle'))
    chart_grid.save(file_path_img+'class_proportion_bl.png', scale_factor = 2.0)

# Function(s)
def p_chart_grid(df, i):
    """
    Creates chart of proportion of observations in 
    each class, gridded by n-gram length

    Parameters
    ----------
    df : DataFrame
        A dataframe with column 'p' for proportion
    
    i : str
        A string to concatenate to the chart title

    Returns
    -------
    A grid of altair charts
    """
    return (alt.Chart(df).mark_bar().encode(
            x = alt.X('p:Q', bin=alt.Bin(step=0.05), title = 'Proportion per class'),
            y = alt.Y("count()")
                ).properties(width = 300, 
                    height = 200, title = "Length "+i))
                
if __name__ == "__main__":
    main(opt["--file_path_data"], 
    opt["--accepted_plates_csv"], 
    opt["--rejected_plates_csv"], 
    opt["--reduced_plate_csv"],
    opt["--X_train_csv"], 
    opt["--y_train_csv"],
    opt["--file_path_img"])
