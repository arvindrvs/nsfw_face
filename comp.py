import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

img = Image.open('check.jpg')
w, h = img.size
imgg=np.array(img)
im = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
Id, conf = recognizer.predict(gray[0:h,0:w])

if(conf>50):
    if(Id==1):
        Id="Arvind"
    elif(Id==2):
        Id="Dillion harper"
    else:
        Id="Unknown"
        
if Id=="Unknown":
    print('No')
else:
    print(Id+' Yes')

print(id,conf)

cv2.waitKey(0)
cv2.destroyAllWindows()
