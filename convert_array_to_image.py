# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:24:29 2018

@author: Fiona Mallett

"""
#Convert numpy array to a picture
from PIL import Image
import os

def convert_to_image(data, name):
    newpath = ('dataset/' + name)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    
    for i in range(0, data.shape[0]):
        img = Image.fromarray(data[i])
        filename = '%s/%s.png' % (newpath, i)
        img.save(filename)
       
