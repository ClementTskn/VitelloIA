#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:40:08 2019

@author: clement
"""
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_FORMATS = ["jpg", "png", "bmp", "ico"]

def myprint(text, verbose):
    if verbose:
        print(text)

def image_generator(directory=".", 
                    rotate = (True, 90), 
                    h_mirror = True, 
                    v_mirror = True,
                    lightness = 0,
                    verbose = True):
    
    '''
    Generates data images from the directory
    Agruments:
        directory : path to the directory (string)
        rotate : (enable, step (degrees))
        h_miror : Horizontal mirror boolean
        v_mirror : Vertical Mirror boolean
        lightness : +- value on each pixel
        verbose : display details in the console
    '''
    # List all the images of the directory
    all_img_list = os.listdir('./'+directory)
    # Only keep the images
    to_delete = []
    for file in all_img_list:
        if file[-3:] not in IMAGE_FORMATS:
            to_delete.append(all_img_list.index(file))
    
    to_delete.sort(reverse=True)
    for i in to_delete:
        all_img_list.pop(i)
    
    myprint(f"{len(all_img_list)} images found in the directory.\n", verbose)
    
    # Openning each image one by one
    for file in all_img_list:
        image = Image.open("./" + file)
        if rotate[0]:
            rotation = rotate[1]
            while rotation < 360-rotate[1]:
                # à continuer ici
                rotation += rotate[1]
        print(type(image))
   
if __name__ == '__main__':
    image_generator()