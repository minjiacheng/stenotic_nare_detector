Use this code to mass crop images for training the stenotic nares model. Cropping based on HOG (Histogram of Oriented Gradient) - high acc but low speed

Modified model from http://blog.dlib.net/2016/10/hipsterize-your-dog-with-deep-learning.html in C++ to make it "python ready"

used dlib (install from whl) to load the dog face detection model (net.dat) and shape prediction model (sp.dat)

Image to be predicted must contain frontal face of the dog

get co-ordinates of the 4 corners of the face bounding box

obtain x y coordinates of left eye, right eye and nose

calculate distance between left and right eye in x direction

create a bounding box around the nose, with width and height equal dist between eyes

crop the bounding box and save the image