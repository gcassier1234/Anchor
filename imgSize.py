# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:40:43 2021

@author: gcass
"""

import numpy as np
import cv2
import glob
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

Lw = []
Lh = [] 
for imgP in glob.glob("boxes/TRUE_TEST/data/*.jpg"):
    print("hého")
    img = cv2.imread(imgP)
    Lw.append(img.shape[0])
    Lh.append(img.shape[1])