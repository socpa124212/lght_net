import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
import datetime
from datetime import timedelta
from rainnet import Data_gathering

T = datetime.datetime(2021,3,7,9,0)

#for i in range(416):
img = Data_gathering(T)
print(img)
    #np.save("/mnt/data/guest1/pro_data/"+T.strftime('%Y%m%d%H%M')+".npy",img)
    #T = T + timedelta(minutes = 2)
