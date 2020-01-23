def count_vectorize_tr_v_te(X_train, X_valid, X_test):
    cv = CountVectorizer(analyzer='char', ngram_range=(2,8))
    X_train_transformed = cv.fit_transform(X_train)
    X_validate_transformed = cv.transform(X_valid)
    X_test_transformed = cv.transform(X_test)
    return X_train_transformed, X_validate_transformed, X_test_transformed