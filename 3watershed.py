#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:20:08 2019

@author: clement.metz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max

import cv2
path = 'x0.75_AetVit.jpg'
#NR_A,B,C.jpg
#x0.75_AetVit.jpg
#NR_AetD.jpg
img = cv2.imread(path)

# Resize
dim = img.shape
newDim = (int(dim[1] / 5), int(dim[0] / 5))
img = cv2.resize(img, dsize=newDim, interpolation=cv2.INTER_CUBIC)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#plt.imshow(gray, cmap='gray')
#plt.show()
normalizedImg = np.zeros(gray.shape)
normalizedImg = cv2.normalize(gray, normalizedImg, 0, 255, cv2.NORM_MINMAX)

for line in range(normalizedImg.shape[0]):
    for i in range(normalizedImg.shape[1]):
        if normalizedImg[line][i] < 10:
            normalizedImg[line][i] = normalizedImg[line][i]*normalizedImg[line][i]
            
#plt.imshow(normalizedImg, cmap='gray')
#plt.show()
ret, thresh = cv2.threshold(normalizedImg,0,1 ,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#plt.imshow(unknown, cmap='gray')
#plt.show()




distance = ndi.distance_transform_edt(sure_fg)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=sure_fg)
markers = ndi.label(local_maxi)[0]


labels = watershed(-distance, markers, mask=sure_fg)
fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(normalizedImg, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap='gray', interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap="gray", interpolation='nearest')
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

###############################################################################
#              ROGNAGE                                                        #
###############################################################################

#Liste des valeurs comprises dans les labels
labels = 255 - labels
valeurs_wat = []
for i in range(255):
    if i in labels:
        if i not in valeurs_wat:
            valeurs_wat.append(i)





















