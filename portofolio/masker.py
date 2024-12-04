import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

faceCascede = cv2.CascadeClassifier("haarcascade_frontalface_alt2 (1).xml")
model= load_model("mask_recog (1).h5")

def face_mask_detector(frame):
    frame = cv2.imread("mas.jpg")
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=faceCascede.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE)

    faces_list=[]
    preds=[]
    for(x,y,w,h) in faces:
        face_frame=frame[y:y+h,x:x+w]
        face_frame=cv2.cvtColor(face_frame,cv2.COLOR_BGR2RGB)
        face_frame=cv2.resize(face_frame,(224,224))
        face_frame=img_to_array(face_frame)
        face_frame=np.expand_dims(face_frame,axis=0)
        face_frame=preprocess_input(face_frame)
        faces_list.append(face_frame)
        face_tensor=np.concatenate(faces_list,axis=0)
        for face in faces_list:
            preds=model.predict(face_tensor)
        for pred in preds:
            (mask, withoutMask)=pred

        label="mask" if mask>withoutMask else"No Mask"
        color=(0,255,0) if label == "mask" else (0,0,255)
        label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        cv2.putText(frame, label,(x,y- 10),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
    cv2.imshow('img',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return frame

face_mask_detector(1)