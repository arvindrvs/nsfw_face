import socket
import cv2
import numpy as np
from PIL import Image

s = socket.socket()          
port = 12345               

s.bind(('', port))         
s.listen(5)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

j=0

while True:
    i=0
    c, addr = s.accept()     

    while True:
        i=i+1
        #print(i)
        txt=c.recv(4096).decode('utf-8')
        #print(txt)

        if txt.startswith("URL "):
            tmp = txt.split(" ")
            urll = tmp[1]
            c.send('GOT URL'.encode())

        else:
            if j==1:
                c.send('Match found. Stop'.encode())
            else:
                c.send('Match not found. Can continue'.encode())
            
            c.close()
            break
        
        myfile = open('server_pic{}.jpg'.format(i),'wb')
        data=c.recv(65536)
        myfile.write(data)
        myfile.close()
        img = Image.open('server_pic{}.jpg'.format(i))
        w, h = img.size
        imgg=np.array(img)
        im = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        Id, conf = recognizer.predict(gray[0:h,0:w])
        #print(id,conf)

        if(conf>50):
            if(Id==1):
                Id="Arvind"
            elif(Id==2):
                Id="Dillion harper"
            else:
                Id="Unknown"
        
            if Id!="Unknown":
                print('Yes'+'\n'+Id+'\n'+urll)
                j=1

cv2.waitKey(0)
cv2.destroyAllWindows()
