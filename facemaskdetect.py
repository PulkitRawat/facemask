import os
import cv2 as cv
from sklearn.svm import SVC
from combined_data import output

cascPathFace = os.path.dirname(cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPathEyes = os.path.dirname(cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv.CascadeClassifier(cascPathFace)
eyeCascade =  cv.CascadeClassifier(cascPathEyes)


#Face detection
##for the webcam
video_capture = cv.VideoCapture(0)

## for the image 
# frame = cv.imread("WhatsApp Image 2023-05-11 at 13.31.53.jpeg")

## for already captured video
# video_capture = cv.VideoCapture("path_to_video")

while True:
    ret, frame = video_capture.read() #only required in case of videos

    # converting to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(60, 60),
                                              flags=cv.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (233, 218, 201), 2)
                        #top-left  bottom-right
        face = frame[y:y+h, x:x+w]
        face = cv.resize(face,(50,50))
        face = face.reshape(1,-1)
        pred = SVC.predict(face)[0]
        n = output(int(pred))
        cv.putText(frame,n,(x,y),fontFace=0,fontScale=1,color=(244,250,250),thickness=2)  
        
        ###eye detection
        # faceROI = frame[y:y+h, x:x+w]
        # eyes  = eyeCascade.detectMultiScale(faceROI)
        # for(x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)  
        #     # the '//' operator represents the result will be rounded of to nearest integer
        #     radius = int(round((w2+h2)*0.25))
        #     cv.circle(frame, eye_center, radius, (255, 221, 173),4)

    cv.imshow('result', frame)
    if cv.waitKey(1) & 0xFF == ord('q'): #escaping the if loop if 'q' is pressed
        break
    
video_capture.release()
cv.destroyAllWindows()