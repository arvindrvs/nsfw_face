import numpy as np
import cv2
from matplotlib import pyplot as plt

import socket
import sys
 
s = socket.socket()         
port = 12345               
s.connect(('127.0.0.1', port))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('porn.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
i=0


for (x,y,w,h) in faces:
    i=i+1
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite('faces{}.jpg'.format(i),crop_img)

    with open('faces{}.jpg'.format(i), 'rb') as f:
        data = f.read()
    size = len(data)
    s.sendall("SIZE {0}".format(size).encode())
    answer = s.recv(4096)
    txt=answer.decode('utf-8')
    if txt == 'GOT SIZE':
        s.sendall(data)
    print('sent')
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

s.sendall('done'.encode())
cv2.imwrite('output.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
