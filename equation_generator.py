''' Student: Marisol Antunez '''

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import invert

import sympy
import cv2

DIRECTORY ='raw_dataset/'

NUMBERS = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATOR = ['add', 'div', 'mul', 'sub']


def crop_image(image, tol=0):
    '''Crops blank space of a 2D image with a tolerance of 0''' 
    image = invert(image)
    mask = image>tol
    
    # Find region of interest
    roi = np.ix_(mask.any(1),mask.any(0))
    cropped_image = image[roi]
    cropped_image = invert(cropped_image)
    
    return cropped_image



def divide_image(im, invert=True):
    '''   ''' 
    original = im.copy()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours, obtain bounding box coordinates, and save all the ROIs coordinates
    contours, _ = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h_list=[]
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if w*h>250:
            h_list.append([x,y,w,h])

    # Sort contours based on horizontal order
    ziped_list=list(zip(*h_list))
    x_list=list(ziped_list[0])
    dic=dict(zip(x_list,h_list))
    x_list.sort()

    
    original = original*255

    # Extract the individial images for all the identified ROIs
    images = []
    for x in x_list:
        [x,y,w,h]=dic[x]
        
        #Retrieve roi image
        roi=im[y:y+h,x:x+w]

        # Preprocess Image so it matches Training set data
        image = rgb2gray(roi)
        image = ((image - np.min(image))/( np.max(image) - np.min(image))) * 255
        image = crop_image(image)
        image = resize(image,(30, 30))
        images.append(image)
        
        # Add ROI rectangle to image
        cv2.rectangle(original, (x, y), (x + w, y + h), (255,0,0), 5)


    plot_roi(original, len(images))

    return images



def plot_roi(image, num):
    '''Plots the ROI image and the number of math symbols identified.
    This was only used to represent and visualy verify the selected ROI by opencv''' 

    image = cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255,255])
    fig = plt.figure()
    fig.patch.set_facecolor('xkcd:light gray')
    plt.imshow(image, cmap='gray')

    # plt.rcParams['axes.facecolor'] = 'black'
    ax = plt.gca()

    # Hide X and Y axes label and tick marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)    
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title('\nROI of Equation', weight="bold")    
    plt.xlabel(str(num) + ' math symbols identified',  weight="bold")
    plt.show()



def plot_equation(img):
    '''Plots a given image equation.
    This was only used to represent and visualy verify the selected equation image''' 

    img = cv2.copyMakeBorder(img, 50,50,50,50, cv2.BORDER_CONSTANT, value=[255,255,255])
    fig = plt.figure()
    fig.patch.set_facecolor('xkcd:light gray')
    plt.imshow(img, cmap='gray')

    # Hide X and Y axes label and tick marks
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    title = '\nRandom Math Equation'
    plt.title(title, weight="bold")    
    plt.show()



def equation_to_image(categories):
    '''From the given math symbol categories it selects a random picture of its folder to represent it.
    The selected images are merge horizontally to have a resulting equation image.

    This was only used when generating a 'random equation' in order have a bigger pool of 'equation images' to test with''' 

    images = []

    # Pic a random image from the file of each category
    for category in categories:
        path = os.path.join(DIRECTORY, category)
        files = os.listdir(path)

        index = random.randint(0, len(files) - 1)
        image = imread(os.path.join(path, files[index]))
        image = resize(image,(200, 200, 3))

        images.append(image)

    # Merge images horizontally
    merge = np.hstack(tuple(images))
    merge = np.float32(merge)
    
    # Save image just in case
    imsave("equation.png", merge)
    final_image_equation = cv2.imread("equation.png")

    plot_equation(final_image_equation)

    return final_image_equation
        


def generate_equation():
    '''Selects a random amount of numbers with random length of digits and appends operator between them to simulate an equation. 

    This was only used to generate a 'random equation' in order have a bigger pool of 'equation images' to test with''' 

    # Select amount of numbers to use for the equation
    numbers = random.randint(2, 4)
    equation = []
    

    for _ in range(0, numbers):
        # If number are already added in the equation, then start adding a random operatr between them
        if equation:
            operator = OPERATOR[random.randint(0, len(OPERATOR) - 1)]
            equation.append(operator)


        # Select a random amount of digits the number should have.
        number_length = random.randint(1, 2)

        # Select the random value of each of those digits
        for i in range(number_length):

            number = NUMBERS[random.randint(0, len(NUMBERS) - 1)]  

            # Avoid starting a number with 0
            while(i==0 and number=='0'):
                number = NUMBERS[random.randint(0, len(NUMBERS) - 1)]  
                
            equation.append(number)

    
    #print(equation)     

    # Make an image with the generated equation
    image = equation_to_image(equation)
    
    return image, equation



def equation_from_file():
    '''Reads a random equation png image from the equations file path''' 

    EQUATIONS_FILE = "my_equations/"
    FILES_TO_SKIP = ['.DS_Store']

    files = os.listdir(EQUATIONS_FILE)    
    
    # Choose and read random file image
    index = random.randint(0, len(files) - 1)
    while(files[index] in FILES_TO_SKIP):
        index = random.randint(0, len(files) - 1)

    image = cv2.imread(os.path.join(EQUATIONS_FILE, files[index]))

    # Reformat Image to a precision allowed by OpenCV
    h, w, _ = image.shape
    image = resize(image,(h, w, 3))
    image = np.float32(image)
    image = image.astype(np.uint8)


    plot_equation(image*255)

    return image



def format_equation(equation_components):
    '''Joins the equation components in a single string and maps the operators to their symbol''' 

    OPERATOR_MAP = { "add":" + ", "div":" / ", "mul":" * ", "sub":" - "}
    equation = ''.join(equation_components)
    
    for i,j in OPERATOR_MAP.items():
        equation = equation.replace(i, j)

    return equation    



def equation_solver(predicted_equation):
    '''Sovles a valid equation with Sympy. If equation is invalid 'Invalid Equation' is returned;
    A predicted_equation could be invalid if the prediction wasn't totally accurate''' 


    #Check equation is valid by veifying it doesn't have 2 math operators together
    for i in range(len(predicted_equation) - 2):
        if((predicted_equation[i] in OPERATOR) and (predicted_equation[i+1] in OPERATOR)): return 'Invalid Equation'


    # Format Equation
    equation = format_equation(predicted_equation)
    
    if(('/ 0' in equation)): return 'Invalid Equation'

    # Solve Equation with Sympy
    equation_sympy = equation + ' - y'
    y = sympy.symbols('y')
    result = sympy.solveset(sympy.sympify(equation_sympy), y)

    if(len(result) == 0): return 'Invalid Equation'

    # Extract result
    result = list(result)[0].n(3)

    return result
