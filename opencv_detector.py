# encoding: utf-8
import numpy as np
import cv2

#cascade = cv2.CascadeClassifier('./opencv_cascade/face/haarcascade_frontalface_default.xml')
#cascade.xml
cascade = cv2.CascadeClassifier('./opencv_cascade/flag/cascade_bk_neg3500/cascade.xml')

img = cv2.imread('./query/pos_query_5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

objects = cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in objects:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
