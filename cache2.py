import subprocess
import os
from pathlib import Path
import glob

os.remove('text.txt')

os.system('E:/ChromeCacheView.exe /scomma D:/check/text.txt')

            
fo = open("text.txt", "r")
while True:
    line = fo.readline()
    if not line: break
    data=line.split(",",2)
    if "data_2  [12288]" in line:
        print(data[0])
        print(data[1])
        break

cmd2 = subprocess.Popen('cmd.exe /C E:\ChromeCacheView.exe /copycache "'+data[1]+'" "image/jpeg" /CopyFilesFolder "D:\check" /UseWebSiteDirStructure 0')
names = glob.glob("D:/Studies/nsfw_test/*.jpg")
for i in names:
    if i.startswith('faces') or i.startswith('server') or i.startswith('output'):
        continue
    img = i
    break


cmd3 = subprocess.Popen('cmd.exe /C python classify_nsfw.py -m data/open_nsfw-weights.npy "'+img+'"')
