# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:22:57 2017

@author: Pavitrakumar
"""

"""
This file is explains how data is colleted and used to training GAN
"""


"""
Data collection: (download images into a folder using below commands)
#curl "https://media.kitsu.io/characters/images/[1-99000]/original.jpg" -o "#1.jpg"
#curl "http://www.anime-planet.com/images/characters/i-[1-60000].jpg" -o "#1.jpg"

We need to extract faces from the above images - we use OpenCV 
and an animeface detector CascadeClassifier xml file (https://github.com/nagadomi/lbpcascade_animeface)
"""

import cv2
import os
from PIL import Image
from tqdm import tqdm

data_dir = "E:\\GAN_Datasets\\curl\\anime_planet\\"
faceCascade = cv2.CascadeClassifier('E:\\Software\\WinPython-64bit-3.5.3.1Qt5\\python-3.5.3.amd64\\Lib\\site-packages\\opencv_python-3.2.0+contrib.dist-info\\lbpcascade_animeface.xml')
output_dir = "E:\\GAN_Datasets\\curl\\ANIME_PLANET_FACES_COLOUR_ONLY_64\\"
file_name = "mk"
crop_size = (64,64)
only_color = True

def biggest_rectangle(r):
    #return w*h
    return r[2]*r[3]

for count,filename in enumerate(tqdm(os.listdir(data_dir))):
    image = cv2.imread(data_dir+filename)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        gray = cv2.equalizeHist(gray)
        # detector options
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor = 1.01,
                                             minNeighbors = 5,
                                             minSize = (90, 90))
        #if any faces are detected, we only extract the biggest detected region
        if len(faces) == 0:
            continue
        elif len(faces) > 1:
            sorted(faces, key=biggest_rectangle, reverse=True)
            
        if only_color and (Image.fromarray(image).convert('RGB').getcolors() is not None):
            continue
            
        x, y, w, h = faces[0]
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_image = image[y:y + h, x:x + w,:]
        resized_image = cv2.resize(cropped_image, crop_size)
        cv2.imwrite(output_dir+str(count)+file_name+".png", resized_image)


"""

#To seperate out color images use .getcolors() - if its a color image, it returns None  --- does not work 100% of the time!! (?)
from PIL import Image
#img = Image.open("E:\\GAN_Datasets\\curl\\ANIME_PLANET_FACES_ALL\\23mk.jpg")
image = cv2.imread("E:\\GAN_Datasets\\curl\\ANIME_PLANET_FACES_ALL\\23mk.jpg")
colors = Image.fromarray(image).convert('RGB').getcolors() #if cv2 open is used
#colors = img.convert('RGB').getcolors() #if PIL open is used
len(colors)


#To re-size existing set of images
data_dir = "E:\\GAN_Datasets\\curl\\ANIME_PLANET_FACES_COLOUR_ONLY_96\\"
output_dir = "E:\\GAN_Datasets\\curl\\ANIME_PLANET_FACES_COLOUR_ONLY_64\\"
crop_size = (64,64)

for count,filename in enumerate(os.listdir(data_dir)):
    image = cv2.imread(data_dir+filename)
    resized_image = cv2.resize(image, crop_size)
    cv2.imwrite(output_dir+filename+".png", resized_image)
"""    
