import numpy as np
import cv2
from matplotlib import pyplot as plt
path = 'NR_AetD.jpg'
img = cv2.imread(path)

# Resize
dim = img.shape
newDim = (int(dim[1] / 5), int(dim[0] / 5))
img = cv2.resize(img, dsize=newDim, interpolation=cv2.INTER_CUBIC)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()
normalizedImg = np.zeros(gray.shape)
normalizedImg = cv2.normalize(gray, normalizedImg, 0, 255, cv2.NORM_MINMAX)
for line in range(normalizedImg.shape[0]):
    for i in range(normalizedImg.shape[1]):
        if normalizedImg[line][i] < 70:
            normalizedImg[line][i] *=10
plt.imshow(normalizedImg, cmap='gray')
plt.show()
ret, thresh = cv2.threshold(normalizedImg,0,1 ,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)




# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
plt.imshow(sure_fg, cmap='gray')
plt.show()
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.imshow(img, cmap='gray')
plt.show()