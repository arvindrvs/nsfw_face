# nsfw_face
Photoshopped and leaked nude photos of celebrities and normal people are circulated throughout the internet. This project is aimed to report all those images (Using Opencv and imagga) to the search engine with image URL and authentication id to remove the content.

# Working
There are 4 main python code files namely app.py, ser.py, trainner.py and dataset.py

First, People who doesnt want their nude content in internet, can signup at a website, which will use webcam to capture their images and store it in dataset (dataset.py)
When the signup process is complete, then the trainner.py is run to train the new dataset and creates a trainer yml file inside trainner folder

In order for this to work, app.py is by default installed in all laptop and computers. Whenever a new image is cached, this program starts to run. app.py acts as a client in communication chanel. The cached image is tested whether it is a NSFW (Not suitable for work) image or not. If the confidence is greater than 80% then the image is scanned for faces. If faces are recognized then that face is cropped out and sent to the server along with its URL

ser.py is installed in a VM in cloud which acts as a server. If any client sends any image, the image is compared to those in the dataset using trainner yml file. If there is a match then the image URL is noted and sent to the search engine along with person's authentication ID to remove that content.

NOTE: Cache reading and scanning the image is not done yet, instead URL is directly pasted.
OpenCV is used for face recognition and comparison while imagga is used for NSFW testing.

# Contributors
Contributions are welcome. Cache reading is still not done and for now the server is run in same system as client. The servers should be moved to cloud and a website must be created to run dataset.py and trainner.py during signup process.

Note: Please pull requests in alternate branch.

# Requirements
Python 3 and above (conda install preferred)
Packages to be installed:
->Numpy
->OpenCV
->PIL/pillow
->requests
An imagga account : https://imagga.com/ (Change api_key (line 11) and api_secret (line 12) in app.py with respect to your account)
OS: Windows (Verified) or linux (should do fine)

Note: This project is still ongoing, and fork it if you are interested.
