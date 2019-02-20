from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import VotingClassifier


def get_ML_model_by_name(algo_name, X, y):
    """
    Returns the model for the chosen algo name

    INPUT:
        algo_name : string        - The name of the algo chosen
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    if (algo_name == 'svm_l2'):
        model = create_linear_SVM_l2(X, y)
    elif (algo_name == 'svm_l1'):
        model = create_linear_SVM_l1(X, y)
    elif (algo_name == 'logistic'):
        model = create_logistic(X, y)
    elif (algo_name == 'ridge'):
        model = create_ridge(X, y)
    elif (algo_name == 'multinomial'):
        model = create_multinomialNB(X, y)
    elif (algo_name == 'bernoulli'):
        model = create_bernoulliNB(X, y)
    else:
        model = create_voting(X, y)

    return model


def train_pipeline(clf, X, y):
    """
    Returns the model for clf trained

    INPUT:
        clf :                     - The classifier to train
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    tvec = TfidfVectorizer().set_params(
        stop_words=None, max_features=100000, ngram_range=(1, 3))

    model_pipeline = Pipeline([('vectorizer', tvec), ('classifier', clf)])
    model_pipeline.fit(X, y)
    return model_pipeline


def create_linear_SVM_l2(X, y):
    """
    Returns the model for linear SVM with parameter l2

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    print('Using linear SVM with parameter l2')
    clf = LinearSVC()
    return train_pipeline(clf, X, y)


def create_linear_SVM_l1(X, y):
    """
    Returns the model for linear SVM with parameter l1

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    print('Using linear SVM with parameter l1')

    clf = Pipeline([('feature_selection',
                     SelectFromModel(LinearSVC(penalty="l1", dual=False))),
                    ('classification', LinearSVC(penalty="l2"))])

    return train_pipeline(clf, X, y)


def create_logistic(X, y):
    """
    Returns the model for Logistic Regression

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    print('Using Logistic Regression')

    clf = LogisticRegression()
    return train_pipeline(clf, X, y)


def create_ridge(X, y):
    """
    Returns the model for Ridge Regression

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    print('Using Ridge Regression')
    clf = RidgeClassifier()
    return train_pipeline(clf, X, y)


def create_multinomialNB(X, y):
    """
    Returns the model for Multinomial Naive Bayes

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    print('Using Multinomial Naive Bayes')

    clf = MultinomialNB()
    return train_pipeline(clf, X, y)


def create_bernoulliNB(X, y):
    """
    Returns the model for Bernoulli Naive Bayes

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    print('Using Bernoulli Naive Bayes')

    clf = BernoulliNB()
    return train_pipeline(clf, X, y)


def create_voting(X, y):
    """
    Returns the model for Voting Classifier using Ridge Classifier,
    Linear SVM and Logistic Regression

    INPUT:
        X : Multidimensional list - The traning features
        y : list                  - The traning results

    OUTPUT:
        Returns the model trained
    """
    print('Using Voting Classifier using Ridge Classifier, Linear SVM and Logistic Regression')

    clf1 = RidgeClassifier()
    clf2 = LinearSVC()
    clf3 = LogisticRegression()

    clf = VotingClassifier(
        estimators=[('rcs', clf1), ('svc', clf2), ('lr', clf3)], voting='hard')
    return train_pipeline(clf, X, y)
