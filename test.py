from rainnet import rainnet
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

in_data = np.zeros((832,928,928,4))
T = datetime.datetime(2021,3,7,9,0)
ext = ".npy"
file_name = T.strftime('%Y%m%d%H%M') + ext
#for i in range(416):
print("in" + T.strftime('%H%M'))
in_data[0,:,:,:] = np.load("/mnt/data/guest1/pro_data/"+file_name)
#T = T + timedelta(minutes= 2)
print(in_data[0,:,:,:])
"""
T = datetime.datetime(2021,3,8,9,0)
file_name = T.strftime('%Y%m%d%H%M') + ext
for i in range(416,832):
    print("in" + T.strftime('%H%M'))
    in_data[i,:,:,:] = np.load("/mnt/data/guest1/pro_data/"+file_name)
    T = T + timedelta(minutes = 2)

im_Str = "/mnt/data/guest1/raw_data/gk2aamile1bwv063ko020lc"
ext = ".srv.png"
ob_data = np.zeros((832,928,928))

T = datetime.datetime(2021,3,7,9,10)
for i in range(416):
    img = plt.imread(im_Str + T.strftime("%Y%m%d%H%M") + ext)
    img = np.pad(img[22:922,:,0],((14,14),(14,14)), mode = "constant", constant_values = 0)
    print("out" + T.strftime('%H%M'))
    ob_data[i,:,:] = img
    T = T + timedelta(minutes = 2)

T= datetime.datetime(2021,3,8,9,10)
for i in range(416,832):
    img = plt.imread(im_Str + T.strftime("%Y%m%d%H%M") + ext)
    img = np.pad(img[22:922, :, 0],((14,14),(14,14)) ,mode = "constant", constant_values = 0)
    print("out" + T.strftime('%H%M'))
    ob_data[i,:,:] = img
    T = T + timedelta(minutes = 2)

print(in_data)
ob_data = tf.expand_dims(ob_data, -1)

in_data = tf.convert_to_tensor(in_data[:800,:,:,:])
ob_data = tf.convert_to_tensor(ob_data[:800,:,:,:])

test_in = in_data[800:832,:,:,:]
test_out = ob_data[800:832,:,:,:]

print(np.shape(in_data))
print(np.shape(ob_data))
print(np.shape(test_in))
print(np.shape(test_out))

model = rainnet()
model.summary()
opt = keras.optimizers.Adam()
model.compile(optimizer = opt,
        loss = 'logcosh',
        metrics = ['logcosh'])

model.fit(in_data, ob_data,
        batch_size = 2,
        epochs = 10,
        verbose = 1)
"""

