def count_vectorize_tr_v_te(X_train, X_valid, X_test, analyzer = 'char', ngram_range = (2,8)):
    """
    Applies CountVectorizer to 3 feature datasets

    Parameters
    ----------
    X_train : df
        Feature training data set

    X_valid : df
        Feature validation data set

    X_test : df
        Feature testing data set

    analyzer : str (optional)
        The analyzer to pass into CountVectorizer. See
        CountVectorizer documentation.
        (default = 'char)

    ngram_range : tuple (optional)
        The range of length n-grams to return. See
        CountVectorizer documentation.
        (default = (2,8))

    Returns
    -------
    3 arrays
        X_train_transformed, X_validate_transformed, X_test_transformed
    """
    cv = CountVectorizer(analyzer='char', ngram_range=(2,8))
    X_train_transformed = cv.fit_transform(X_train)
    X_validate_transformed = cv.transform(X_valid)
    X_test_transformed = cv.transform(X_test)
    return X_train_transformed, X_validate_transformed, X_test_transformed