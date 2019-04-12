from PIL import Image
import glob
import numpy as np
import os
from os.path import basename

image_list = []
for filename in glob.glob('Dataset/*.png'): #assuming png
    im=Image.open(filename)
    label = int(filename.split("_")[1])
    #print(label)
    image_list.append(im)
