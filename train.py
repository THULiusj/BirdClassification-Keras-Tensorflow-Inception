# Train the data with InceptionV3 and transfer learning
# Use real-time data augmentation in Keras

import pandas
import numpy as np
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils.training_utils import multi_gpu_model

IM_SIZE = (299, 299) #fixed size for InceptionV3
NB_EPOCHS = 200 #epoch number
BAT_SIZE = 64 #batch size
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
GPU_NUM = 1
NB_CLASSES = 200

# data prep
train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
)
test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'images',
    target_size=IM_SIZE,
    batch_size=BAT_SIZE
)

validation_generator = test_datagen.flow_from_directory(
    'test_images',
    target_size=IM_SIZE,
    batch_size=BAT_SIZE
)
# setup model
base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
temp_model = GlobalAveragePooling2D()(base_model.output)
temp_model = Dense(FC_SIZE, activation='relu')(temp_model) #new FC layer, random init
predictions = Dense(NB_CLASSES, activation='softmax')(temp_model) #new softmax layer
model = Model(inputs=base_model.input, outputs=predictions)
#model = multi_gpu_model(model, gpus=GPU_NUM)

#nb_train_samples = "train_images"
#nb_val_samples = "test_images"
# transfer learning
#for layer in base_model.layers:
#    layer.trainable = False
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# transfer learning
for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
history_tl = model.fit_generator(
    generator=train_generator,
    epochs=NB_EPOCHS,
    steps_per_epoch=len(train_generator),
    verbose=2
)

# save the model
model.save('bird.hdf5')
