# # https://youtu.be/nbRkLE2fiVI
# https://youtu.be/1HqjPqNglPc
"""


Dataset from: http://press.liacs.nl/mirflickr/mirdownload.html

Read high res. original images and save lower versions to be used for SRGAN.

Here, we are resizing them to 128x128 that will be  used as HR images and
32x32 that will be used as LR images
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

train_dir = "." #dataset vako location hala hai ani tyo folde lai original_images name deu

for img in tqdm(sorted(os.listdir(train_dir + "/original_images")), desc="Preprocessing image: "):
    img_array = cv2.imread(train_dir + "/original_images/" + img)
    try:
        img_array = cv2.resize(img_array, (128,128))
        lr_img_array = cv2.resize(img_array,(32,32))
        cv2.imwrite(train_dir+ "/hr_images/" + img, img_array)
        cv2.imwrite(train_dir+ "/lr_images/"+ img, lr_img_array)
    except:
        print(f"error in: {img}")
