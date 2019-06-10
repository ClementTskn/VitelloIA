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
    myprint(all_img_list, verbose)
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
        name = "./" + directory + "/" + file
        image = Image.open(name)
        rotation = rotate[1]
        while rotation <= 360-rotate[1]:
            # Rotations
            if rotate[0]:
                new_image = image.rotate(rotation)
                # Saving transformed image
                new_image.save(name+"_r=" + str(rotation)+".png", format="PNG")
            else:
                new_image = image
            
            # Mirror
            if h_mirror:
                # Saving transformed image
                new_image.transpose(Image.FLIP_TOP_BOTTOM).save(name+"_r=" + str(rotation)+"_hmir"+".png", format="PNG")
            if v_mirror:
                # Saving transformed image
                new_image.transpose(Image.FLIP_LEFT_RIGHT).save(name+"_r=" + str(rotation)+"_vmir"+".png", format="PNG")
            if v_mirror and h_mirror:
                # Saving transformed image
                new_image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(name+"_r=" + str(rotation)+"_hvmir"+".png", format="PNG")
            
            if rotate[0]:
                rotation += rotate[1]
            else:
                rotation = 360
   
if __name__ == '__main__':
    image_generator('photos/cellules classées/vitreg/Regression amas', rotate=(True, 20), h_mirror=True, v_mirror=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    