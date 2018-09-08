[![HitCount](http://hits.dwyl.io/arvindrvs/nsfw_face.svg)](http://hits.dwyl.io/arvindrvs/nsfw_face)
[![Build Status](https://travis-ci.org/arvindrvs/nsfw_face.png?branch=master)](https://travis-ci.org/arvindrvs/nsfw_face)

# nsfw_face
Photoshopped and leaked nude photos of celebrities and normal people are circulated throughout the internet. This project is aimed to report all those images (Using Opencv and tensorflow) to the search engine with image URL and authentication id to remove the content.

# Working
There are 5 main python code files namely cache2.py, nsfw.py, ser.py, trainner.py and dataset.py

First, People who doesnt want their nude content in internet, can signup at a website, which will use webcam to capture their images and store it in dataset (dataset.py)
When the signup process is complete, then the trainner.py is run to train the new dataset and creates a trainer yml file inside trainner folder

In order for this to work, nsfw.py and cache2.py are by default installed in all laptop and computers. Whenever a cached image is updated or about to be deleted, cache2.py program starts to run. To be deleted image is given as input by default system. Then, cache2.py creates a text file and stores all the information of all the caches and searches the cache to be deletd along with the URL and checks whether it is an image or not (Its because the ChromeCacheView API doesn't have option of saving directly using cache name). If it is an image, then that particular file is saved and it calls nsfw.py.

Note: nsfw (Not suitable for work) check ((i.e.) to check whether the given image contains nudity or not) is yahoo opensouce project, and it was converted from caffe to tensorflow by Marc Dietrichstein.

nsfw.py tests whether the cached image is a NSFW (Not suitable for work) image or not. If the confidence is greater than 80% then the image is scanned for faces. If faces are recognized then that face is cropped out and sent to the server along with its URL

ser.py is installed in a VM in cloud which acts as a server. If any client sends any image, the image is compared to those in the dataset using trainner yml file. If there is a match then the image URL is noted and sent to the search engine along with person's authentication ID to remove that content.

OpenCV is used for face recognition and comparison while yahoo nsfw using tensorflow is used for NSFW testing.

# Contributors
Contributions are welcome. The server is run in same system as client, for now. The servers should be moved to cloud and a website must be created to run dataset.py and trainner.py during signup process.

Note: Please pull requests in alternate branch.

# Requirements
Python 3.6 and above (conda install preferred)  
Tensorflow 1.x (GPU prefered)

Packages to be installed:  
->Numpy  
->OpenCV  
->PIL/pillow  
->requests

ChromeCacheView : https://www.nirsoft.net/utils/chrome_cache_view.html  
Tensorflow implementation of open NSFW model : https://github.com/mdietrichstein/tensorflow-open_nsfw

OS: Windows (Verified) or linux (Didn't check)  
Browser: Only chrome (for now)
Images : Only jpeg (for now)

Note: This project is still ongoing, fork and contribute it if you are interested.
