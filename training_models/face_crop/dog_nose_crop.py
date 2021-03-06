# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:38:46 2018

@author: A53445
"""

import os
import dlib
import numpy as np
import PIL

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

directory = './source_imgs'
crop_dir = './cropped_imgs'
detector = dlib.cnn_face_detection_model_v1('net.dat') #load saved dog face detection model
predictor = dlib.shape_predictor('sp.dat') #load saved dog facial key point predictor
crop_width = 150 #size of the cropped image

n=1
for f in os.listdir(directory):
    print("Processing file: {}".format(f))
    img = load_image_file(os.path.join(directory,f), mode='RGB') #load image as 8-bit rgb
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
        image_to_crop = PIL.Image.open(os.path.join(directory,f)).convert('RGB')
        crop_area = (co_ord[0]-dist, co_ord[1]-dist, co_ord[0]+dist, co_ord[1]+dist)
        print('cropping around: ', crop_area)
        cropped_image = image_to_crop.crop(crop_area)
        crop_size = (crop_width, crop_width)
        cropped_image.thumbnail(crop_size)
        img_name = 'crop_img'+str(n)+'.jpg'
        cropped_image.save(os.path.join(crop_dir,img_name))
    n += 1
