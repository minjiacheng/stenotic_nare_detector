# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:38:46 2018

@author: A53445
"""

import dlib
import numpy as np
import PIL
from ensemble import load_models, ensemble
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

detector = dlib.cnn_face_detection_model_v1('net.dat') #load saved dog face detection model
predictor = dlib.shape_predictor('sp.dat') #load saved dog facial key point predictor
v1_model, v2_model, Xception_model, class_list = load_models() #load saved model

def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)

def crop_img(img_path):
    #use face feature detector to locate nose and crop around it
    img = load_image_file(img_path, mode='RGB') #load image as 8-bit rgb
    dets = detector(img, 1) #upsample image once then get prediction from model 
    print("Number of faces detected: {}".format(len(dets)))
    
    for i, d in enumerate(dets):
        #print co-ordinates of the bounding box
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
        shape = predictor(img, d.rect)
            
        reye = str(shape.part(5)).split()
        reye = int(reye[0][1:][:-1])
        leye = str(shape.part(2)).split()
        leye = int(leye[0][1:][:-1])
        dist = int(abs(reye-leye)/2)
            
        print('dist_between_eye is:', dist)
        print("Nose: {}".format(shape.part(3)))
        temp = str(shape.part(3)).split()
        co_ord = [int(temp[0][1:][:-1]),int(temp[1][:-1])]
        #start cropping image
        image_to_crop = PIL.Image.open(img_path)
        crop_area = (co_ord[0]-dist, co_ord[1]-dist, co_ord[0]+dist, co_ord[1]+dist)
        print('cropping around: ', crop_area)
        cropped_image = image_to_crop.crop(crop_area)
        crop_size = (150, 150)
        cropped_image.thumbnail(crop_size)
        cropped_image.save('./images/cropped_image.jpg')

def make_prediction(img_path):
    prediction = ensemble(v1_model, v2_model, Xception_model, class_list, img_path)
    #plot images
    original = mpimg.imread(img_path)
    cropped = mpimg.imread('./images/cropped_image.jpg')

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cropped)
    plt.title("I predict this dog has: " + str(prediction))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#to make another prediction, run code below this line
img_path = './images/images.jpg' #path to the image to predict
cropped_image = crop_img(img_path)
make_prediction(img_path)