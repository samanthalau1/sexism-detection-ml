# methods for evaluating model performance

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from yellowbrick.classifier import ConfusionMatrix, ClassPredictionError


# outputs various statistics about model performance
def print_stats(X_test_engtweets, y_test_engtweets, model):
    y_pred = model.predict(X_test_engtweets)
    print('Accuracy:', accuracy_score(y_test_engtweets, y_pred))
    print('Precision:', precision_score(y_test_engtweets, y_pred, pos_label='sexist'))
    print('Recall:', recall_score(y_test_engtweets, y_pred, pos_label='sexist'))
    print('F1 score:', f1_score(y_test_engtweets, y_pred, pos_label='sexist'))


# graph of confusion matrix
def conf_matrix(X_train_engtweets, y_train_engtweets, X_test_engtweets, y_test_engtweets, model):
    classes = ['sexist', 'non-sexist']
    cm = ConfusionMatrix(
        model, classes=classes,
        percent=True, label_encoder={0: 'sexist', 1: 'non-sexist'})

    cm.fit(X_train_engtweets, y_train_engtweets)
    cm.score(X_test_engtweets, y_test_engtweets)

    cm.show();

    for label in cm.ax.texts:
        label.set_size(22)


# graph of class prediction error
def predict_error(X_train_engtweets, y_train_engtweets, X_test_engtweets, y_test_engtweets, model):
    classes = ['sexist', 'non-sexist']
    visualizer = ClassPredictionError(
    model, classes=classes)
    visualizer.fit(X_train_engtweets, y_train_engtweets)
    visualizer.score(X_test_engtweets, y_test_engtweets)
    visualizer.show();