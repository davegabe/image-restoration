import numpy as np
import cv2 as cv
import os

def corrupt(training_path, evaluation_path, training_corrupted_path, evaluation_corrupted_path):
    """
    Corrupt training and evaluation data and save it to the specified destinations.
    
    Args:
        training_path: The path to the training data.
        evaluation_path: The path to the evaluation data.
        training_corrupted_path: The path to save the corrupted training data.
        evaluation_corrupted_path: The path to save the corrupted evaluation data.
    """

    for img in os.listdir(training_path):
        draw(img)

        file_name = os.path.basename(img)
        new_name = os.path.splitext(file_name)[0]+"_corrupted"+os.path.splitext(file_name)[1]

        final_path = os.path.join(training_corrupted_path, new_name)
    
        cv.imwrite(final_path, img)

    for img in os.listdir(evaluation_path):
        draw(img)

        file_name = os.path.basename(img)
        new_name = os.path.splitext(file_name)[0]+"_corrupted"+os.path.splitext(file_name)[1]

        final_path = os.path.join(evaluation_corrupted_path, new_name)
    
        cv.imwrite(final_path, img)



def draw(img):
    
    x = img.shape[0]
    y = img.shape[1]
    xlimit = x/3.5
    ylimit = y/3.5
    
    for i in range(np.random.randint(3,6)):            
        
        upper_edge = np.random.randint(low=(0,0), high=min(x,y))
        lower_edge = np.random.randint(low=upper_edge+5, high=(upper_edge[0]+5+xlimit,upper_edge[1]+5+ylimit))
        
        cv.rectangle(img, upper_edge, lower_edge, color=(0,0,0), thickness = -1 )
