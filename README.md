# nsfw_face
Photoshopped and leaked nude photos of celebrities and normal people are circulated throughout the internet. This project is aimed to report all those images (Using Opencv and imagga) to the search engine with image URL and authentication id to remove the content.

# Working
There are 5 main python code files namely cache.py, app.py, ser.py, trainner.py and dataset.py

First, People who doesnt want their nude content in internet, can signup at a website, which will use webcam to capture their images and store it in dataset (dataset.py)
When the signup process is complete, then the trainner.py is run to train the new dataset and creates a trainer yml file inside trainner folder

In order for this to work, app.py and cache.py are by default installed in all laptop and computers. Whenever a cached image is updated or about to be deleted, cache.py program starts to run. To be deleted image is given as input by default system. Then, cache.py creates a text file and stores all the information of all the caches and searches the cache to be deletd along with the URL and checks whether it is an image or not (Its because the ChromeCacheView API doesn't have option of saving directly using cache name). If it is an image, then that particular file is saved.

Note: I didn't generate nsfw (Not suitable for work) check ((i.e.) to check whether the given image contains nudity or not) code. Instead an online API named imagga is used for now. In future an offline nsfw code will be generated.  

app.py acts as a client in communication chanel. The cached image is tested whether it is a NSFW (Not suitable for work) image or not. If the confidence is greater than 80% then the image is scanned for faces. If faces are recognized then that face is cropped out and sent to the server along with its URL

ser.py is installed in a VM in cloud which acts as a server. If any client sends any image, the image is compared to those in the dataset using trainner yml file. If there is a match then the image URL is noted and sent to the search engine along with person's authentication ID to remove that content.

NOTE: Cache is read and image is saved, but imagga (used for nsfw check) will accept only URL. 
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

ChromeCacheView : https://www.nirsoft.net/utils/chrome_cache_view.html  
An imagga account : https://imagga.com/ (Change api_key (line 11) and api_secret (line 12) in app.py with respect to your account)

OS: Windows (Verified) or linux (Didn't check)  
Browser: Only chrome (for now)

Note: This project is still ongoing, fork and contribute it if you are interested.
