import cv2
import sys
import logging as log
import datetime as dt
import os
import re
import numpy as np
import shelve,random
from time import sleep

faceCascade = cv2.CascadeClassifier("Face_cascade.xml")
log.basicConfig(filename='webcam.log',level=log.INFO)
FONT = cv2.FONT_HERSHEY_DUPLEX
video_capture = cv2.VideoCapture(0)
anterior = 0

Datafile = shelve.open("Data")
if 'Data' not in Datafile.keys():
    Datafile['Data']=list()
    Data_list = list()
else:
    Data_list = Datafile["Data"]

def Make_Changes(label):
    if label not in Data_list:
        Data_list.append(label)

def get_images(path):
    images = list()
    labels = list()
    count=0
    if len(os.listdir(path)) == 0:
        print "Empty Dataset.......aborting Training"
        exit()
    for img in os.listdir(path):
        regex = re.compile(r'(\d+|\s+)')
        labl = regex.split(img)
        labl = labl[0]
        count=count+1
        Make_Changes(labl)
        image_path =os.path.join(path,img)
        image=cv2.imread(image_path)
        image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        images.append(image_gray)
        labels.append(Data_list.index(labl))
    return images,labels,count

def initialize_recognizer():
    face_recognizer = cv2.createLBPHFaceRecognizer()
    print "Training.........."
    Dataset = get_images("./Dataset")
    print "Recognizer trained using Dataset: "+str(Dataset[2])+" Images used"
    face_recognizer.train(Dataset[0],np.array(Dataset[1]))
    return face_recognizer

face_r = initialize_recognizer()

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #image=cv2.VideoCapture.grab(frame)
    image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image_grey,scaleFactor=1.2,minNeighbors=5,minSize=(25,25),flags=0)
    temp_set = list()
    face_list = list()
    for x,y,w,h in faces:
        sub_img=image_grey[y:y+h,x:x+w]
        img=image[y:y+h,x:x+w]
        temp_set.append(img)
        nbr,conf = face_recognizer.predict(sub_img)
        print "confidence", conf
        face_list.append([nbr,conf]);
        cv2.rectangle(image,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),2)
        cv2.putText(image,Data_list[nbr],(x,y-10), FONT, 0.5,(0,0,255),1)
        cv2.putText(image,str(conf),(x,y+h+20), FONT, 0.5,(0,0,255),1)

    Datafile['Data']=Data_list
    Datafile.close()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('k-michael', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
