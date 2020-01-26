# L1c3nc3 t0 C0d3  
**DCSI_522_group-415**  
Authors: Keanna Knebel, Cari Gostic, Furqan Khan

## Project Proposal

### Dataset  
For this project, we'll be using data on applications for vanity license plates submitted to the NYDMV. This dataset is a result of a Freedom of Information Act (FOIA) request by the WYNC public radio station and therefore is within the public domain.  It covers vanity license plate applications from 10/1/2010 to 9/26/2014. The dataset is hosted on github [here.](https://github.com/datanews/license-plates)  

### Research Question  
In this project, we will attempt to identify patterns in the existing ambiguity of vanity plate cancellations. Specifically, we will create a classification model to answer the following predictive questions:
>- **What features are the strongest predictors of a rejected license plate?**
>- **Can you predict if a vanity plate that passes all rules (laid out in the cancellation procedure and red-guide) will be accepted or rejected by the NYS DMV?**

### EDA
The dataset set contains two classifications of `plate` configurations, accepted and rejected. A useful visualization that we created in our EDA was a bar graph comparing the counts of observations for each class. The analysis showed that there is significant class imbalance. Out of 133,636 total examples, only 1646 belong to the rejected class, which is only 1.23% of the total examples. This is important to note, because not accounting for this imbalance will likely result in a model that achieves a very high overall score, but performs very poorly when predicting the rejected class.  

### Analysis Plan  
We will be performing our analysis in Python to take advantage of the scikit-learn package. We will use sklearn's CountVectorizer funciton to engineer features from each plate. The features will be character strings of varying lengths (n-grams). We will use sklearn's GridSearchCV to optimize the length of n-grams used (i.e. 2,3 and 4 letter strings, 4,5 and 6 letter strings, etc.). Then, we'll fit a MultinomialNB model to the training split of the data and evaluate model performance. Once we feel the model is optimized, we can identify the strongest predictors of rejected license plates using the predict_proba attribute of the fit model.

We are also exploring ways to account for the significant class imbalance. One method we're considering is to undersample the accepted class.

### Report plan
We plan to display a subset of the strongest features in a bar graph, and are considering adding a word map to enhance this visualization. We can also show a confusion matrix to asses the efficacy of our model in performing its prescribed task.
