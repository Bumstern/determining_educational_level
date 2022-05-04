import matplotlib.pyplot as plt
from sklearn import svm, metrics, model_selection
import numpy as np


def main():
    # Create train and test samples
    embeds = np.load('embeds/embeds.npy')
    labels = np.load('embeds/labels.npy')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(embeds, labels, test_size=0.2, random_state=42)

    # Create SVM classifier
    classifier = svm.SVC(C=1, probability=True)
    classifier.fit(X_train, y_train)

    # Get predictions
    pred = classifier.predict(X_test)
    proba = classifier.predict_proba(X_test)
    print(pred)
    print(y_test)

    # Show metrics
    fpr, tpr, _ = metrics.roc_curve(y_test, proba[:,0], pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)
    print('roc_auc: ', roc_auc)
    acc = metrics.accuracy_score(y_test, pred)
    print('acc: ', acc)
    f1 = metrics.f1_score(y_test, pred, average='binary')
    print('f1: ', f1)

    # Plot ROC-AUC curve
    plt.style.use('seaborn')
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    main()
