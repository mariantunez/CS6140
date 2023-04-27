''' Student: Marisol Antunez '''

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from equation_generator import generate_equation, divide_image, equation_solver, format_equation, equation_from_file


def plot_results(image, predicted, result):
    '''Plots the equation image with the predicted equation and equation result''' 
    title = "Predicted Equation = " + str(predicted)
    title += "\nFormatted Equation = " + str(format_equation(predicted))

    image = cv2.copyMakeBorder(image*255,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255,255])

    fig = plt.figure()
    fig.patch.set_facecolor('xkcd:light gray')

    plt.imshow(image, cmap='gray')
    plt.title(title)    
    plt.xlabel('Result = ' + str(result), weight="bold")

    # Hide X and Y axes label and tick marks
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    

def main():
    '''Runs the entire pipeline of the Handwritten Equation Solver with SVM+PCA''' 
    
    X_TRAIN_CSV = 'dataset/x_train.csv'
    Y_TRAIN_CSV = 'dataset/y_train.csv'
    x_train = pd.read_csv(X_TRAIN_CSV)
    y_train = pd.read_csv(Y_TRAIN_CSV) 
    print(pd.DataFrame(x_train))

    # Get Random Equation
    # equation_image, _ = generate_equation()  # Generated Equation from dataset
    equation_image = equation_from_file() # Handwritten equation
    
    # Divide Equation into individual images
    equation_divided = divide_image(equation_image)

    # Flatten Images
    data_to_predict = []
    for im in equation_divided:
        image = im.flatten()
        image = np.int_(image)
        data_to_predict.append(image)
    print(pd.DataFrame(data_to_predict))

    # Apply PCA
    pca = PCA(0.85).fit(x_train)
    x_train = pca.transform(x_train)
    data_to_predict = pca.transform(data_to_predict)
    

    # Train Model
    model = SVC(kernel='rbf', C=10)
    model.fit(x_train, np.array(y_train).ravel())

    # Predict
    y_pred = model.predict(data_to_predict)    
    print('Predicted Result = ' + str(y_pred))  


    # Solve Equation
    solution = equation_solver(y_pred)
    plot_results(equation_image, y_pred, solution)

main()    