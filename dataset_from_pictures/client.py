import cv2
from PIL import Image
import requests
import numpy as np
from io import BytesIO
import socket
import sys

url='https://i.pinimg.com/736x/56/60/ba/5660ba6529611cee23af24c608312f2b--motion-sexy-girls.jpg'
response = requests.get(url)
imgg = np.array(Image.open(BytesIO(response.content)))
img = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)


s = socket.socket()         
port = 12345               
s.connect(('127.0.0.1', port))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
    
s.sendall('done'.encode())
cv2.imwrite('output.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
