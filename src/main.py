# full execution of sexism detection model

from data_utils import load_data, clean_cols, clean_text, remove_stop, use_eng, stem_and_lemm, tfidf_extract, word2vec_extract, split_y
from plots import lang_plot, sexism_plot, count_sexist, count_nonsexist, sexist_wordcloud, nonsexist_wordcloud
from train import train_log_model, tune, train_clf_model
from evaluation import print_stats, conf_matrix, predict_error

# import and clean data
train_data, test_data = load_data()

# some visualizations
lang_plot(train_data)
sexism_plot(train_data)
count_sexist(train_data)
count_nonsexist(train_data)

# more text cleaning
train_clean_data, test_clean_data = clean_cols(train_data, test_data)
train_clean_data, test_clean_data = clean_text(train_clean_data, test_clean_data)
train_clean_data, test_clean_data = remove_stop(train_clean_data, test_clean_data)

# using english tweets only
train_engtweets, test_engtweets = use_eng(train_clean_data, test_clean_data)

# wordcloud visualizations
sexist_wordcloud(train_engtweets)
nonsexist_wordcloud(train_engtweets)

train_engtweets, test_engtweets = stem_and_lemm(train_engtweets, test_engtweets)

# different feature engineering techniques

# tfidf
X_train_engtweets, X_test_engtweets = tfidf_extract(train_engtweets, test_engtweets)

# word2vec
#X_train_engtweets, X_test_engtweets = word2vec_extract(train_engtweets, test_engtweets)

y_train_engtweets, y_test_engtweets = split_y(train_engtweets, test_engtweets)

# training logistic regression model and hyperparameter tuning
model = train_log_model(X_train_engtweets, y_train_engtweets)
#tune(X_train_engtweets, y_train_engtweets, model, 'logistic')

# evaluation for logistic regression
print_stats(X_test_engtweets, y_test_engtweets, model)
conf_matrix(X_train_engtweets, y_train_engtweets, X_test_engtweets, y_test_engtweets, model)
predict_error(X_train_engtweets, y_train_engtweets, X_test_engtweets, y_test_engtweets, model)

# training random forest classifier model and hypterparameter tuning
clf = train_clf_model(X_train_engtweets, y_train_engtweets)

# evaluation for clf
print_stats(X_test_engtweets, y_test_engtweets, clf)
conf_matrix(X_train_engtweets, y_train_engtweets, X_test_engtweets, y_test_engtweets, clf)
predict_error(X_train_engtweets, y_train_engtweets, X_test_engtweets, y_test_engtweets, clf)