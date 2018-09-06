# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:16:11 2018

@author: A53445
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ensemble import load_models, ensemble

#load all the models
v1_model, v2_model, Xception_model, class_list = load_models()

def make_prediction(img_path):
    #make prediction
    final_result = ensemble(v1_model, v2_model, Xception_model, class_list, img_path)
    #plot image
    nose = mpimg.imread(img_path)
    plt.imshow(nose)
    plt.title("I predict this dog has: " + str(final_result))
    plt.axis('off')
    plt.figure()

#to make another prediction, run code below this line
img_path = 'data/test/y06.jpg' #path to image
make_prediction(img_path)