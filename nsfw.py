#!/usr/bin/env python
"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

import os
import sys
import argparse
import glob
import time
from PIL import Image
from StringIO import StringIO
import caffeimport requests
import cv2
from PIL import Image
import requests
import numpy as np
from io import BytesIO
import socket




def resize_image(data, sz=(256, 256)):
    """
    Resize image. Please use this resize logic for best results instead of the 
    caffe, since it was used to generate training dataset 
    :param str data:
        The image data
    :param sz tuple:
        The resized image dimensions
    :returns bytearray:
        A byte array with the resized image
    """
    img_data = str(data)
    im = Image.open(StringIO(img_data))
    if im.mode != "RGB":
        im = im.convert('RGB')
    imr = im.resize(sz, resample=Image.BILINEAR)
    fh_im = StringIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)
    return bytearray(fh_im.read())

def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
    output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        img_data_rs = resize_image(pimg, sz=(256, 256))
        image = caffe.io.load_image(StringIO(img_data_rs))

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = max((H - h) / 2, 0)
        w_off = max((W - w) / 2, 0)
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                    **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input file.
    parser.add_argument(
        "input_file",
        help="Path to the input image file"
    )

    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file."
    )

    args = parser.parse_args()
    image_data = open(args.input_file).read()

    # Pre-load caffe model.
    nsfw_net = caffe.Net(args.model_def,  # pylint: disable=invalid-name
        args.pretrained_model, caffe.TEST)

    # Load transformer
    # Note that the parameters are hard-coded for best results
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net.blobs['data'].data.shape})
    caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # Classify.
    scores = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=['prob'])

    # Scores is the array containing SFW / NSFW image probabilities
    # scores[1] indicates the NSFW probability
    print "NSFW score:  " , scores[1]


main(sys.argv)
flag=0

try:
    for x in range(0,3):
        if scores[1] >= 80.0:
                flag=1
except:
    pass

print(flag)

if flag==1:
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
        s.sendall("URL {0}".format(url).encode())
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
