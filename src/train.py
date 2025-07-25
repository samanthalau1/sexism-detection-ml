# methods for creating and training the ML model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# training logistic regression model on the data
def train_log_model(X_train_engtweets, y_train_engtweets):
    model = LogisticRegression(
        C=1.0, penalty='l2', solver='liblinear', random_state=1
    )
    model.fit(X_train_engtweets, y_train_engtweets)
    return model


# training random forest classifier model on the data
def train_clf_model(X_train_engtweets, y_train_engtweets):
    clf = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
    clf.fit(X_train_engtweets, y_train_engtweets)
    return clf


# hyperparameter tuning for logistic regression
def tune(X_train_engtweets, y_train_engtweets, model, type):

    # possible params
    if(type == 'logistic'):
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        grid = dict(solver=solvers, penalty=penalty, C=c_values)
    elif(type == 'rand forest'):
        n_estimators = [10, 100, 1000]
        max_features = ['sqrt', 'log2']
        grid = dict(n_estimators=n_estimators, max_features=max_features)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train_engtweets, y_train_engtweets)

    # print best results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))