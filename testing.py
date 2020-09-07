import os
import sys
import glob
import cv2
from pathlib import Path

# path = Path(r'C:\Users\hoffie\Documents\GitHub\yolov5\inference\images')
path = Path(r'C:\Users\hoffie\Pictures\cropping-template')

if not os.path.exists((path.joinpath("cropped"))):
    os.makedirs((path.joinpath("cropped")))

cropped_path = Path(path, 'cropped')
# test_image = 2020-05-11_07-50-38_006.jpg

def isBigger (biggest, current):
    if biggest[3] > current[3] and biggest[4] > current[4]:
        return False
    else:
        if biggest[3] * biggest[4] > current[3] * current[4]: # area check
            return False
        return True

# open all of the image's text files and look at all bounding boxes
# Each row is class, x_center, y_center, width, and height format
# Ex: 0 0.472083 0.500388 0.405833 0.668478
for file in path.glob('*.txt'):
    with file.open() as f:
        biggest = (0,0,0,0,0)

        for line in f:
            temp = tuple(map(float,line.split())) # typcast to a float
            if isBigger(biggest, temp):
                biggest = temp # if returned true temp is now the biggest
        biggest = biggest + (file.stem,) # append the file name without extension to the tuple
        
        # read image
        img = cv2.imread(str(path.joinpath(str(biggest[5]) + '.jpg')))
        dimensions = img.shape # return size of image (height, width, channels) note: opencv is y,x for most things

        # denormalize the coridantes for this image
        x_cen = int(biggest[1] * dimensions[1]) # normalized x-cen * image width
        y_cen = int(biggest[2] * dimensions[0]) # normalized y-cen * image height
        width = int(biggest[3] * dimensions[1]) # normalized width * image width
        height = int(biggest[4] * dimensions[0]) # normalized height * image height

#cv2.imshow("original", img) # show the original
#v2.waitKey(0) # wait until we press a key

padding = 50

x_start = x_cen - int(width / 2) - padding
x_end = x_cen + int(width / 2) + padding

y_start = y_cen - int(height / 2) - padding
y_end = y_cen + int(height / 2) + padding


if x_start < 0:
    x_start = 0
if x_end > dimensions[1]:
    x_end = dimensions[1] - 1
if y_start < 0:
    y_start = 0
if y_end > dimensions[0]:
    y_end = dimensions[0] - 1


cropped = img[y_start:y_end, x_start:x_end].copy() # y,x
#cv2.imshow("cropped", cropped)
#cv2.waitKey(0)

cv2.imwrite(str(cropped_path.joinpath(str(biggest[5]) + '.jpg')) , cropped)