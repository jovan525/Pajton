import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
img_rgb1 = cv.imread('board.png')
img_rgb2 = cv.imread('Screenshot_5.png')

#img=cv.resize(img_rgb2,(90,90))

img_gray1 = cv.cvtColor(img_rgb1, cv.COLOR_BGR2GRAY)
img_gray2 = cv.cvtColor(img_rgb2, cv.COLOR_BGR2GRAY)

#cv.imwrite('belipesak.png',img)

w, h = img_gray2.shape[::-1]
 
res = cv.matchTemplate(img_gray1,img_gray2,cv.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb1, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
 
cv.imwrite('res.png',img_rgb1)

print(loc)