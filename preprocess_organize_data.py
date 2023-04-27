''' Student: Marisol Antunez '''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
from skimage.io import imread, imsave
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.transform import resize

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


NUMBERS = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATOR = ['add', 'div', 'mul', 'sub']


DIRECTORY ='raw_dataset/' 
PROCESS_DIRECTORY ='dataset/' 
PROCESS_DIRECTORY2 ='processed_dataset/' 

X_TRAIN_CSV = PROCESS_DIRECTORY + 'x_train.csv'
Y_TRAIN_CSV = PROCESS_DIRECTORY + 'y_train.csv'
X_TEST_CSV = PROCESS_DIRECTORY + 'x_test.csv'
Y_TEST_CSV = PROCESS_DIRECTORY + 'y_test.csv'


images=[] #input array
target=[] #output array


def crop_image(image, tol=0):
    '''Crops blank space of a 2D image with a tolerance of 0''' 
    image = invert(image)
    mask = image>tol
    
    # Find region of interest
    roi = np.ix_(mask.any(1),mask.any(0))
    cropped_image = image[roi]
    cropped_image = invert(cropped_image)
    
    return cropped_image



def preprocess_images(categories, cropLimit=200, cropImage=True):
    '''Reads the images from a given file path. ''' 
    total = 0
    saved = 0

    for category in categories:
        path = os.path.join(DIRECTORY, category)

        for img in os.listdir(path):
            total += 1

            # Read Image and transform to gray
            image = imread(os.path.join(path,img))
            image = resize(image,(200, 200, 3))
            image = rgb2gray(image)
            

            # Normalize Image
            image = ((image - np.min(image))/( np.max(image) - np.min(image))) * 255

            # Remove blank soroundings of image
            if(cropImage):            
                image = crop_image(image)

                # If crop wasn't succesful skip image
                _ , width = np.array(image).shape
                if(width >= cropLimit): continue
            saved += 1

            # Resize images to same dimension after cropping
            image = resize(image, (30, 30))

            # Save images
            path2 = os.path.join(PROCESS_DIRECTORY2, category)
            path2 = os.path.join(path2,img)
            imsave(path2, image)    

            # Collapse image to 1-D
            image = image.flatten()
            image = np.int_(image)
                
            # Save image and its target value
            images.append(image)
            target.append(category)

                
        percentage = (saved/total)*100        
        percentage = np.round(percentage, 1)
        print(f'loaded category: {category}, kept {percentage}% -> {saved} images')



def shuffle_dataset(dataset):
    '''Shiffles the rows of given dataset and resets its index''' 
    dataset = shuffle(dataset, random_state=1)
    dataset.reset_index(inplace=True, drop=True)

    return dataset


def write_csv(dataset, filename):
    '''Write a dataset to a csv with given filename''' 
    dataset.to_csv(filename, header=True, index=False, mode='w')


def main():

    # Read and preprocess images    
    preprocess_images(NUMBERS, 100)
    preprocess_images(OPERATOR, 150)

    # Merge images and target values into dataframe
    img=np.array(images)
    trgt=np.array(target)

    dataset = pd.DataFrame(img) 
    dataset['Target']=trgt


    # Shuffle rows of dataset
    dataset = shuffle_dataset(dataset)
    x = dataset.iloc[:,:-1] # Images
    y = dataset.iloc[:,-1]  # Target values
    print(dataset)
    print(x)
    print(y)

    # Split training and test sets and save in csv file
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, test_size=0.20, random_state=2)

    # write_csv(x_train, X_TRAIN_CSV)
    # write_csv(x_test, X_TEST_CSV)
    # write_csv(y_train, Y_TRAIN_CSV)
    # write_csv(y_test, Y_TEST_CSV)
    

main()