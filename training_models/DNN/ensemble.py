import cv2
import numpy as np
import csv, os
from functools import partial, update_wrapper
from keras.applications.xception import preprocess_input
from keras.applications.xception import Xception
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adadelta
from keras.models import Sequential, Model

def model_from_scratch_v1():
    # dimensions of our images.
    img_width, img_height = 64, 64
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    def wrapped_partial(func, *args, **kwargs):
    	partial_func = partial(func, *args, **kwargs)
    	update_wrapper(partial_func, func)
    	return partial_func
    
    def binary_crossentropy_weigted(y_true, y_pred, class_weights):
    	y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    	loss = K.mean(class_weights*(-y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)),axis=-1)
    	return loss
    
    custom_loss = wrapped_partial(binary_crossentropy_weigted, class_weights=np.array([1.0, 2.0]))
    #custom loss function that penalises predicting stenosis as no stenosis
    model.compile(optimizer=Adadelta(), loss=[custom_loss], metrics=['accuracy'])
    return model

def model_from_scratch_v2():
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 1), activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    def wrapped_partial(func, *args, **kwargs):
    	partial_func = partial(func, *args, **kwargs)
    	update_wrapper(partial_func, func)
    	return partial_func
    
    def binary_crossentropy_weigted(y_true, y_pred, class_weights):
    	y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    	loss = K.mean(class_weights*(-y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)),axis=-1)
    	return loss
    
    custom_loss = wrapped_partial(binary_crossentropy_weigted, class_weights=np.array([1.0, 2.0]))
    #custom loss function that penalises predicting stenosis as no stenosis
    classifier.compile(optimizer='adam', loss=[custom_loss], metrics=['accuracy'])
    return classifier

def load_class_list(class_list_file):
    class_list = []
    with open(class_list_file, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            class_list.append(row)
    class_list.sort()
    return class_list

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) # New softmax layer
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

def load_models():
    v1_model = model_from_scratch_v1()
    v1_model.load_weights('models/weight_v1.h5')
    v2_model = model_from_scratch_v2()
    v2_model.load_weights('models/weight_v2.h5')
    class_list_file = "models/Xception_stenosis_class_list.txt"
    class_list = load_class_list(class_list_file)    
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    Xception_model = build_finetune_model(base_model, 1e-3, [1024, 1024], len(class_list))
    Xception_model.load_weights('models/Xception_model_weights.h5')
    return v1_model, v2_model, Xception_model, class_list

def ensemble(v1_model, v2_model, Xception_model, class_list, img_path):
    #read images in grayscale for model v1 & v2
    img = cv2.imread(img_path,0)
    img = cv2.resize(img, (64,64))
    img = img.reshape(1, 64, 64, 1)
    #predictions
    v1_prediction = float(v1_model.predict(img)[0])
    v1_prediction = [1-v1_prediction, v1_prediction]
    v2_prediction = float(v2_model.predict(img)[0])
    v2_prediction = [1-v2_prediction, v2_prediction]
        
    #load images in colour for Xception
    image = cv2.imread(img_path)
    image = np.float32(cv2.resize(image, (128,128)))
    image = preprocess_input(image.reshape(1, 128, 128, 3))
    #predictions
    out = Xception_model.predict(image)[0]
    out = (out + v1_prediction + v2_prediction)/3    
    print(out)
    class_prediction = list(out).index(max(out))
    class_name = class_list[class_prediction]
    return class_name[0]

def test_performance():
    v1_model, v2_model, Xception_model, class_list = load_models()
    x_dir = 'data/validation/stenotic_nares'
    y_dir = 'data/validation/no_stenotic_nares'
    err_cnt_nare_as_none = 0
    err_cnt_none_as_nare = 0
    for f in os.listdir(x_dir):
        img_path = os.path.join(x_dir, f)
        final_result = ensemble(img_path)
        if final_result != 'stenosis':
            err_cnt_nare_as_none+=1
    for f in os.listdir(y_dir):
        img_path = os.path.join(y_dir, f)
        final_result = ensemble(img_path)
        if final_result != 'no_stenosis':
            err_cnt_none_as_nare+=1
    print(err_cnt_nare_as_none)
    print(err_cnt_none_as_nare)
    acc = (100-(err_cnt_nare_as_none+err_cnt_none_as_nare))/100
    print(acc)