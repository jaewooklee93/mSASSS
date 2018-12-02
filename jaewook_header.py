# python internal
import os
import re
import zipfile
from glob import glob
from math import floor, ceil
from os.path import *

# matplotlib
import matplotlib
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['font.size'] = 20
import matplotlib.pyplot as plt

# data processing
import numpy as np
import pandas as pd
import cv2
from skimage.io import *
from skimage.color import *
from skimage.transform import *

# deep learning
from keras.models import Sequential, Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Flatten, Dense
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras import optimizers
import tensorflow as tf

# Visualization
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def visualize(model):
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

def imshowx(img,h,w):
    fig, ax=plt.subplots(h,w,figsize=(4*w,4*h))
    fig.set_facecolor('white')
    ax=ax.ravel()
    for i,a in enumerate(ax):
        a.axis('off')
        if i<img.shape[0]:
            a.imshow(img[i])
    #fig.tight_layout()
    fig.show()
    

# def overlay(img,mask):
#     z=np.zeros_like(img)
#     return cv2.addWeighted(np.dstack((img,img,img)),.5,np.dstack((z,z,mask)),.5,0)

def overlay(images,masks):
    images2=np.concatenate((images,images,images),axis=3)
    if masks.ndim==4 and masks.shape[3]==1:
        zeros=np.zeros_like(masks)
        masks2=np.concatenate((masks,zeros,zeros),axis=3)
    else:
        masks2=masks
    return np.clip(images2+masks2,0,1)



def level(mask):
    if mask.ndim==4:
        return np.array([level(mask[i]) for i in range(mask.shape[0])])
    else:
        sv=mask[:,:,0]
        h=np.sum(np.array([(i-1)*mask[:,:,i]/8 for i in range(1,8)]),axis=0)%1
        return hsv2rgb(np.dstack((h,sv,sv)))


def level2(mask):
    """
    mask:   (H, W, 8)
    return: (H, W, 3)
    """
    if mask.ndim==4:
        return np.array([level2(mask[i]) for i in range(mask.shape[0])])
    
    import matplotlib.cm
    cmap = matplotlib.cm.get_cmap('jet')
    h = np.linspace(0,1,7)
    h = cmap(h)
    h = h[:,np.newaxis,:3]
    h = rgb2hsv(h)
    h[:,:,1]=1
    h = hsv2rgb(h)
    h = np.flip(h, axis=0)
    
    return np.sum(
        np.array([
            np.dstack([mask[:,:,i+1]]*3)*
            h[i]
            for i in range(7)
        ]),axis=0
    )
    
def imshow2(images,rows=0,cols=5):
    '''
    images: (B, H, W, 1 or 3)
    '''
    if rows==0:
        rows=ceil(images.shape[0]/cols)
    fig,ax=plt.subplots(rows,cols,figsize=(cols*5,rows*5))
    fig.set_facecolor('white')
    ax=ax.ravel()
    for i,a in enumerate(ax):
        a.axis('off')
        if i<images.shape[0]:
            a.imshow(np.squeeze(images[i]))
    fig.show()

    
def overlay2(image, mask):
    """
    image:  (H, W, 1) tensor
    mask:   (H, W, 8) tensor
    return: (H, W, 3) tensor
    or batch of (image, masks)
    """
    if image.ndim == 4:
        return np.array([overlay2(image[i], mask[i]) 
                         for i in range(image.shape[0])])
    
    alpha = np.dstack([mask[:,:,0]]*3)
    image = np.dstack([image] * 3)
    mask = level2(mask)
    
    return np.clip(image * (1 - alpha) + mask * alpha, 0, 1)

seed=42