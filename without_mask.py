import os
import cv2 as cv
import numpy as np

cascPathFace = os.path.dirname(cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPathEyes = os.path.dirname(cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv.CascadeClassifier(cascPathFace)
eyeCascade =  cv.CascadeClassifier(cascPathEyes)

#Collecting data
## without mask
video_capture = cv.VideoCapture(0)
data = [] 
while True:
    ret, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(60, 60),
                                              flags=cv.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (233, 218, 201), 2)
        face = frame[y:y+h, x:x+w,:]
        face = cv.resize(frame,(50,50))
        print(len(data))
        if len(data)<2000:
            data.append(face)
            
    cv.imshow("result",frame)
    if cv.waitKey(2)==27 or len(data) >=2000:
        break
video_capture.release()
cv.destroyAllWindows()

np.save("withoutmask.npy", data)
wom = np.load("withoutmask.npy") 
wom = wom.reshape(2000,50*50*3)