import numpy as np
import cv2

fc=cv2.CascadeClassifier("face.xml")
pr = cv2.imread("sai.jpg")
g=cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY)

face=fc.detectMultiScale(g,1.02,10)
print(face)

for(x,y,w,h) in face:
    cv2.rectangle(pr,(x,y),(x+w,y+h),(0,255,0),4)
    cv2.imshow("img",pr)
    cv2.waitKey(0)
cv2.destroyAllWindows()