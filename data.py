from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
import datetime
from datetime import timedelta

T = datetime.datetime(2021,3,7,9,0)
img_string = "/mnt/data/guest1/raw_data/gk2aamile1bwv063ko020lc"
ext = '.srv.png'
dataset = np.zeros([928,928,4])
for i in range(4):
    time = T.strftime('%Y%m%d%H%M')
    img = img_string + time + ext
    sate_img = mpimg.imread(img,cv2.IMREAD_GRAYSCALE)
    padded_img = np.pad(sate_img[22:922,:,0], ((14,14),(14,14)), mode="constant", constant_values=0)
    dataset[:,:,i] = padded_img
    T = T + timedelta(minutes = 2)
    
plt.imshow(dataset[:,:,0])
