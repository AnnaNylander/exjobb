import cv2
import os
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Create video from png pictures. RUN IN PYTHON 3.5!! ')
parser.add_argument('--path', metavar='path',
                    dest='path', default='',
                    help='Path to folder where images is stored. The WHOLE path is needed.')
parser.add_argument('--filename', metavar='name', dest='name', default='img_',
                    help='Prefix of name (before the number). Eg. \'img_\' in \'img_1.png\'. (default \'img_\')')
                    
args = parser.parse_args()

image_folder = args.path

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('video.avi', , 10.0, (500,420)) #(640,480)

images = []
lenght = len(os.listdir(image_folder))
for f in range(0,lenght):
    frame = cv2.imread(image_folder + 'img_%i.png' %f)
    images.append(frame)
    frame = np.array(frame)
    video.write(frame)

video.release()
cv2.destroyAllWindows()
