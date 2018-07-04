import sys
import argparse
import tensorflow as tf
import os
import glob
import time
from PIL import Image
from io import StringIO
import requests
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import socket

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="Path to the input image.\
Only jpeg images are supported.")
    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")
    parser.add_argument("-u", "--url", required=True,
                        help="URL of the nude image")

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",
                        default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    model = OpenNsfwModel()

    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        sess.run(tf.global_variables_initializer())

        image = fn_load_image(args.input_file)

        predictions = \
            sess.run(model.predictions,
                     feed_dict={model.input: image})

        print("Results for '{}'".format(args.input_file))
        print("\tNSFW score:\t{}".format(predictions[0][1]))


main(sys.argv)
flag=0

if predictions[0][1] >= 70.0:
    flag=1

print(flag)

if flag==1:
    img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


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
        s.sendall("URL {0}".format(args.url).encode())
        answer = s.recv(4096)
        txt=answer.decode('utf-8')
        if txt == 'GOT URL':
            s.sendall(data)
        print('sent')
        
    print('sending')
    s.sendall('done'.encode())
    answer=s.recv(10240)
    print(answer.decode('utf-8'))
    cv2.imwrite('output.jpg',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
