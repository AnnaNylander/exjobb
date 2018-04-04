import cv2
import os
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Create input and output files from recorded data')
parser.add_argument('--path', metavar='path',
                    dest='path', default='/media/annaochjacob/crucial/datasets/test_set1/images/',
                    help='Path to folder where images is stored.')

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
