Aim to make a neural network that can identify dogs with stenotic nares. 
Proof of concept project to see the nuances we may encounter if we decide to productionise this in future.
This condition can be used to identify diseases such as BOAS which constitutes a large portion of the pet claims.
Currently insuring dogs with high risk or already possessing stenotic nares but we can't tell unless it's elicitly stated in medical records
Easier to ask owner to send face image.

Ask user for a frontal face image of the dog with nostril clearly visible.
Crop down to just nose.
Turn into grey scale.

Training models based on https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
used custom loss function to penalise predicting stenosis as no stenosis
trained with 400 images from google and validated with 100. All cropped down to just nose either by hand or by dlib model automatically.
For training from scratch, used image augmentation to increase number of training images.

Train from scratch v1 gives about 81% acc
pipline:
turn images to grayscale
stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers
then two fully-connected layers with a 50% drop out rate
end model with a single unit and sigmoid activation
weights saved to 'weight_v1.h5'

Train from scratch v2 has approx 80% acc
a simpler version of v1 with no drop out
weights saved as weight_v2.h5

Very confident in its predictions even when it's wrong. >3times more likely to misclassify no_nares as nares
From scratch models used custom loss function so must compile the model again before loading weights. All scratch models are saved in model.py

Transfer learn with Xception from https://github.com/GeorgeSeif/Transfer-Learning-Suite
Accuracy 81%

Ensemble the models together by voting - which class has the higher number of votes will be the final prediction
Voting because confidence level between models are inconsistent 
Final overall accuracy 83%, of which model classifies stenosis class with 94% acc