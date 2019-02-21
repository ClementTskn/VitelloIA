import numpy as np
import cv2
import matplotlib.pyplot as plt

image = 'x0.75_A.jpg'
img = cv2.imread(image, 1)
img_orig = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (21,21), cv2.BORDER_DEFAULT)

plt.imshow(img, cmap='gray')
plt.show()
all_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9, 120, param1 = 50, param2 = 30)
print(all_circles)
print(all_circles.shape)

all_circles_rounded = np.uint16(np.around(all_circles))
print(all_circles_rounded)
print(all_circles_rounded.shape)
print("There are " + str(all_circles_rounded.shape[1]) + " cells")

count = 1
for i in all_circles_rounded[0, :]:
    cv2.circle(img_orig, (i[0], i[1]), i[2], (50, 200, 200), 5)
    cv2.circle(img_orig, (i[0], i[1]), 2, (0, 0, 255), 3)
    count+=1

plt.imshow(img_orig)
plt.show()
