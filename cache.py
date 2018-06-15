import subprocess
import os
from pathlib import Path

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
