import cv2
from PIL import Image
import requests
import numpy as np
from io import BytesIO

url='https://i.pinimg.com/736x/56/60/ba/5660ba6529611cee23af24c608312f2b--motion-sexy-girls.jpg'
response = requests.get(url)
img = np.array(Image.open(BytesIO(response.content)))

cv2.imwrite('porn.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
