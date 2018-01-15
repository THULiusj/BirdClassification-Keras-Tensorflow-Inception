# test the trained model with internet image

from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
import urllib.request as urllib2
import io
import sys
from keras.preprocessing import image
import matplotlib.pyplot as plt

# load model
model = load_model("bird.hdf5")  
print('load model complete')
imglist = np.genfromtxt('classes.txt',delimiter=' ',dtype='str')

# load image
fd = urllib2.urlopen(sys.argv[1])
image_file = io.BytesIO(fd.read())
im = image.load_img(image_file, target_size = (299,299))
im = image.img_to_array(im)
im = np.expand_dims(im, axis = 0)
im /= 255
print("load image complete")
#plt.imshow(im[0])
#plt.show()

# predict image    
predictions = model.predict(im)
print('RESULT: ', imglist[np.argmax(predictions),1])
