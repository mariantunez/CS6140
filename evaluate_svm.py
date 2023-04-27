
''' Student: Marisol Antunez '''

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report


X_TRAIN_CSV = 'dataset/x_train.csv'
Y_TRAIN_CSV = 'dataset/y_train.csv'
X_TEST_CSV = 'dataset/x_test.csv'
Y_TEST_CSV = 'dataset/y_test.csv'

x_train = pd.read_csv(X_TRAIN_CSV)
y_train = pd.read_csv(Y_TRAIN_CSV)  
x_test = pd.read_csv(X_TEST_CSV)
y_test = pd.read_csv(Y_TEST_CSV) 

LABELS = {'0','1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'mul', 'sub'}
KERNEL = ['rbf', 'poly', 'linear']
REGULARIZATION = [0.1, 1, 10, 100]
PCA_VARIANCE = [0.85, 0.90, 0.95, 0.99]


def plot_confusion_matrix(expected, predicted, title, labels=None):
    '''Plots Confussion Matrix of predicted dataset''' 
    matrix = confusion_matrix(expected, predicted)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    display.plot()

    plt.title('Confusion Matrix \n' + title, weight="bold")
    plt.show()



def get_accuracy(expected, predicted, model_title):
    '''Print the accurary of a model''' 
    accuracy = accuracy_score(predicted, expected)
    accuracy = np.round(accuracy, 3)
    print(f"{model_title} ---> {accuracy*100}% accurate")



def evaluate_svm_pca(kernel, c):
    '''Evaluate SVM with the given kernel and C parameter plus variations with PCA''' 
    title = f"SVC(kernel={kernel}, C={c})"

    # Evaluate Model accuracy without PCA
    model = SVC(kernel=kernel, C=c)
    model.fit(x_train, np.array(y_train).ravel())
    y_pred = model.predict(x_test)
    get_accuracy(y_test, y_pred, title)

    # Evaluate model accuracy with PCA for variances [0.85, 0.90, 0.95, 0.99]
    for variance in PCA_VARIANCE:
        if((kernel == 'linear') and (variance==0.85)): continue  # Skip combination as it never runs

        pca = PCA(variance).fit(x_train)
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)

        # Predict after applying pca
        model.fit(x_train_pca, np.array(y_train).ravel())
        y_pred = model.predict(x_test_pca)

        title_pca = title + f" + PCA({variance})"
        get_accuracy(y_test, y_pred, title_pca)



def performance_report(kernel, c, pca_variance):
    '''Prints Classification Report and Plots Confussion Matrix for a SVC with a given kernel, C and PCA variance''' 
    pca = PCA(pca_variance).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    model = SVC(kernel=kernel, C=c)
    model.fit(x_train_pca, np.array(y_train).ravel())
    y_pred = model.predict(x_test_pca)

    title = f"SVC(kernel={kernel}, C={c}) + PCA({pca_variance})"
    print("\u0332".join(title))
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=3))
    plot_confusion_matrix(y_test, y_pred, title)



def main():
    # Evaluate SVM accuracyw with each combination of kernel and regularization parameters
    for kernel in KERNEL:
        for c in REGULARIZATION:
            evaluate_svm_pca(kernel, c)


    # ----- Best Performing Model was "SVC(kernel='rbf', C=10) + PCA(0.85)"  ------
    # Print Classification Report  and Confusion Matrix for best performing model
    performance_report('rbf', 10, 0.85)


main()
