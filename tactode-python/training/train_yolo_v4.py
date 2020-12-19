# -*- coding: utf-8 -*-
"""Yolov4CustomTactode.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lwVg6mR9QlxNTUDadsHVFGiJRaxtz4z_
"""

# Commented out IPython magic to ensure Python compatibility.
# Connect to drive
from google.colab import drive
drive.mount('/content/drive')
# Change directory to colab folder
# %cd /content/drive/My\ Drive/Colab\ Notebooks/tactode/yolov4/darknet/

# Commented out IPython magic to ensure Python compatibility.
# Create new folder named yolov4
# %mkdir yolov4

# Clone darknet repo inside new folder & get pre-trained weights
# %cd yolov4
!git clone https://github.com/AlexeyAB/darknet.git
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp" -O yolov4.conv.137 && rm -rf /tmp/cookies.txt

# Commented out IPython magic to ensure Python compatibility.
# Compile darknet with opencv and GPU support
# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
print("Building. . . It might take 2-3 minutes")
!make &> build_log.txt

# Commented out IPython magic to ensure Python compatibility.
#Create a custom cfg file 
# %cp cfg/yolov4-custom.cfg cfg/yolov4-obj.cfg
#Change .cfg file according to https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

#Create obj.names file
!touch data/obj.names
!echo "end_forever" >> data/obj.names
!echo "end_repeat" >> data/obj.names
!echo "forever" >> data/obj.names
!echo "repeat" >> data/obj.names
!echo "flag" >> data/obj.names
!echo "lower_a" >> data/obj.names
!echo "lower_d" >> data/obj.names
!echo "lower_e" >> data/obj.names
!echo "lower_i" >> data/obj.names
!echo "lower_s" >> data/obj.names
!echo "backward" >> data/obj.names
!echo "forward" >> data/obj.names
!echo "left" >> data/obj.names
!echo "right" >> data/obj.names
!echo "0" >> data/obj.names
!echo "1" >> data/obj.names
!echo "2" >> data/obj.names
!echo "3" >> data/obj.names
!echo "4" >> data/obj.names
!echo "5" >> data/obj.names
!echo "6" >> data/obj.names
!echo "7" >> data/obj.names
!echo "9" >> data/obj.names
!echo "addition" >> data/obj.names
!echo "division" >> data/obj.names
!echo "answer" >> data/obj.names
!echo "question" >> data/obj.names
!echo "erase" >> data/obj.names
!echo "pen_down" >> data/obj.names
!echo "pen_up" >> data/obj.names

#Create obj.data file
!touch data/obj.data
!echo "classes = 32" >> data/obj.data
!echo "train = data/train.txt" >> data/obj.data
!echo "valid = data/test.txt" >> data/obj.data
!echo "names = data/obj.names" >> data/obj.data
!echo "backup = backup/" >> data/obj.data

# Commented out IPython magic to ensure Python compatibility.
#Download the dataset
# %cd /content/drive/My\ Drive/Colab\ Notebooks/tactode/yolov4/darknet/data
# %ls
!unzip /content/drive/My\ Drive/Colab\ Notebooks/tactode/yolov4/darknet/data/train.zip -d /content/drive/My\ Drive/Colab\ Notebooks/tactode/yolov4/darknet/data/obj/
# !rm dataset.zip
#%cd ..

import cv2
from os import chdir, getcwd
import glob
from bs4 import BeautifulSoup


def get_files(directory, extension):
  files = []
  actualDir = getcwd()
  # print(actualDir)
  chdir(directory)
  for ext in extension:
    for f in glob.glob(ext):
      files.append(f)
      # print(f)
  chdir(actualDir)
  return files

def get_bbox(obj):
  xmin = int(obj.find('xmin').text)
  ymin = int(obj.find('ymin').text)
  xmax = int(obj.find('xmax').text)
  ymax = int(obj.find('ymax').text)

  return [xmin, ymin, xmax, ymax]


def get_label(obj):
  if obj.find('name').text == "with_mask":
    return 1
  elif obj.find('name').text == "mask_weared_incorrect":
    return 1
  return 0


def convert2YOLOLabel(coords, image):
  height, width, c = image.shape
  xc = (coords[0] + (coords[2] - coords[0]) / 2) / width
  yc = (coords[1] + (coords[3] - coords[1]) / 2) / height
  w = (coords[2] - coords[0]) / width
  h = (coords[3] - coords[1]) / height
  # print(coords, width, height, xc*width, yc*height)
  # print(xc, yc, w, h)
  return [xc, yc, w, h]

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My\ Drive/Colab\ Notebooks/tactode/yolov4/darknet/data
#%ls
#%cat train.txt

path_to_dataset = '/content/drive/My Drive/Colab Notebooks/tactode/yolov4/darknet/data/obj/train/'
imageFiles = get_files(path_to_dataset, ["*.jpg"])
path = 'data/obj/train/'

print(len(imageFiles))

with open('data/train.txt', 'w') as outputfile:
  for img in imageFiles:
    outputfile.write(path+img)
    outputfile.write('\n')
  outputfile.close()

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My\ Drive/Colab\ Notebooks/tactode/yolov4/darknet
!chmod +x darknet
#!./darknet detector train data/obj.data cfg/yolov4-obj.cfg ../yolov4.conv.137 -dont_show
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_last.weights -dont_show

# !cp cfg/yolov4-obj-mask.cfg ../
# !cp backup/yolov4-obj-mask_final.weights ../
# !cp data/obj.names ../