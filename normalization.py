# -*- coding: utf-8 -*-
"""
Functions for image normalization

Created on Wed Apr 29 2020
@author: Shihan Qiu (qiushihan19@gmail.com)
"""

import numpy as np

# -----------  ----------- 
# subject-wise normalization
def normalize_case_zscore(image, mask):
    mask_img = image*mask
    ind = np.nonzero(mask_img)
    image_nonzero = mask_img[ind]
    norm_image = (image - np.mean(image_nonzero)) / np.std(image_nonzero)
    return norm_image

def normalize_case_mean(image, mask):
    mask_img = image*mask
    ind = np.nonzero(mask_img)
    image_nonzero = mask_img[ind]
    norm_image = image / np.mean(image_nonzero) # normalization by dividing the mean value
    return norm_image

def normalize_case_minmax(image):
    norm_image = (image - np.amin(image))/ (np.amax(image) - np.amin(image))
    return norm_image

# slice-wise normalization
def normalize_slc_mean(image, mask, bkgmask):
    # mask: for normalization
    # bkgmask: to remove background
    norm_image = np.empty(np.shape(image))
    for i in range(np.shape(image)[2]):
        image_slc = image[:,:,i]
        mask_slc = mask[:,:,i]
        if np.sum(mask_slc) == 0:
            norm_image[:,:,i] = image_slc
        else:
            ind = np.nonzero(mask_slc)
            norm_image[:,:,i] = image_slc / np.mean(image_slc[ind])
    norm_image = np.squeeze(norm_image)
    norm_image = norm_image*bkgmask  # set background as 0
    return norm_image
    
# dividing by 95 percentile slice by slice
def normalize_slc_95prc(image, mask, bkgmask):
    # mask: for normalization
    # bkgmask: to remove background
    norm_image = np.empty(np.shape(image))
    for i in range(np.shape(image)[2]):
        image_slc = image[:,:,i]
        mask_slc = mask[:,:,i]
        if np.sum(mask_slc) == 0:
            norm_image[:,:,i] = image_slc
        else:
            ind = np.nonzero(mask_slc)
            norm_image[:,:,i] = image_slc / np.percentile(image_slc[ind],95)
    norm_image = np.squeeze(norm_image)
    norm_image = norm_image*bkgmask  # set background as 0
    return norm_image

# slice-wise normalization for Multitasking spatial factors
def normalize_U0_slc_95prc(U0,mask,bkgmask):
    # mask: for normalization
    # bkgmask: to remove background
    norm_U0 = np.empty(np.shape(U0))
    for i in range(np.shape(U0)[2]):
        U0_1ch_slc = np.squeeze(U0[:,:,i,0])
        mask_slc = mask[:,:,i]
        if np.sum(mask_slc) == 0:
            norm_U0[:,:,i,:] = U0[:,:,i,:]
        else:
            ind = np.nonzero(mask_slc) 
            norm_U0[:,:,i,:] = U0[:,:,i,:] / np.percentile(U0_1ch_slc[ind],95)
    norm_U0[:,:,:,2:5] = norm_U0[:,:,:,2:5]*10
    mask_ch = np.tile(bkgmask,(np.shape(U0)[3],1,1,1))
    mask_ch = np.moveaxis(mask_ch, 0, -1)
    norm_U0 = np.multiply(norm_U0, mask_ch)  # set background as 0
    return norm_U0

def normalize_U0_slc_mean(U0,mask,bkgmask):
    # mask: for normalization
    # bkgmask: to remove background
    norm_U0 = np.empty(np.shape(U0))
    for i in range(np.shape(U0)[2]):
        U0_1ch_slc = np.squeeze(U0[:,:,i,0])
        mask_slc = mask[:,:,i]
        if np.sum(mask_slc) == 0:
            norm_U0[:,:,i,:] = U0[:,:,i,:]
        else:
            ind = np.nonzero(mask_slc) 
            norm_U0[:,:,i,:] = U0[:,:,i,:] / np.mean(U0_1ch_slc[ind])
    norm_U0[:,:,:,2:5] = norm_U0[:,:,:,2:5]*10
    mask_ch = np.tile(bkgmask,(np.shape(U0)[3],1,1,1))
    mask_ch = np.moveaxis(mask_ch, 0, -1)
    norm_U0 = np.multiply(norm_U0, mask_ch)  # set background as 0
    return norm_U0
    
