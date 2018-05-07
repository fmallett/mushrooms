# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:11:21 2018

@author: Fiona Mallett
Student Number: 3289339
"""

from keras.models import Sequential  #used to initialise CNN
from keras.layers import Convolution2D #adding conoltuion layers
from keras.layers import MaxPooling2D # add pooling layers
from keras.layers import Flatten #flattening
from keras.layers import Dense #add fully connected layers

#Initialising the CNN
classifier  = Sequential()

#Step 1: Convolution - feature dector - find features in image
 #paramteters of convolution layer.. number of filters, rows and columns
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))  #good practice to start with 32, then 64, 128..

#Step 2: Max Pooling - stride of 2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3: Flattening -into 1D vector
classifier.add(Flatten())

#Step 4: Full Connection
#add hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu')) #no. of nodes in hidden layer (between no of nodes in input and output)
#output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #sigmoid for binary output

#Compile the CNN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Fit CNN to the input -- from keras documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', #this will be a folder in our directory
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary') #2 classes is binary eg cat and dog

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)


