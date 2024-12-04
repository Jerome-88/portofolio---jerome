
import cv2
import numpy as np

img=cv2.imread("bangun.jpg")
cv2.imshow('img',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray",gray)

#thresh
ret,thresh= cv2.threshold(gray,50,255,1)
cv2.imshow("thresh",thresh)

contours,h=cv2.findContours(thresh,1,2)#memberi informasi mengenai bentuk

for c in contours:
    approxpolydp=cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    n = len(approxpolydp)

    if n==6:
        cv2.drawContours(img,[c],0,(255,0,0),5)
    elif n==3:
        cv2.drawContours(img,[c],0,(0,255,255),8)
    elif n>9:
        cv2.drawContours(img,[c],0,(200,150,200),10)
    elif n==4:
        cv2.drawContours(img,[c],0,(221,221,123),5)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np

img=cv2.imread("motor.jpg")
cv2.imshow('img',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray",gray)

#thresh
ret,thresh= cv2.threshold(gray,50,255,1)
cv2.imshow("thresh",thresh)

contours,h=cv2.findContours(thresh,1,2)#memberi informasi mengenai bentuk

for c in contours:
    approxpolydp=cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    n = len(approxpolydp)

    if n==4:
        cv2.drawContours(img,[c],0,(221,221,123),5)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
import cv2
image=cv2.imread("motor.jpg")
import imutils

image=imutils.resize(image, width=500)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

(T,threshinv) = cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV,cv2.THRESH_OTSU)
cv2.imshow("threshinv",threshinv)

cnts=cv2.findContours(threshinv.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[0]
clone=image.copy

for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    if(w<500 and w>300 and h<150 and h>70):
        print(x,y,w,h)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
        cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''