import cv2
import numpy as np
from PIL import Image

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
for i in range(1,21):
    img = Image.open('{}.jpg'.format(i))
    imgg=np.array(img)
    im = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        crop_img = gray[y:y+h, x:x+w]
        cv2.imwrite('User.2.{}.jpg'.format(i),crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
