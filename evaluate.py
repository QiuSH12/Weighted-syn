# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 2020
@author: Shihan Qiu (qiushihan19@gmail.com)
"""

import os

# select GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "6,0"

import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
import io
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity


# parameters to set
IMG_SIZE = 256
IN_CHANNEL = 5
OUT_CHANNEL = 3
BATCH_SIZE = 4

from normalization import normalize_U0_slc_95prc, normalize_slc_95prc

# evaluation for one cv group
def eval_1cvgroup(test_folder,test_id, checkpoint_path, result_path):
    # ----------- load data -----------
    X_test = np.empty([0,IMG_SIZE,IMG_SIZE,IN_CHANNEL])
    Y_test = np.empty([0,IMG_SIZE,IMG_SIZE,OUT_CHANNEL])
    mask_test = np.empty([0,IMG_SIZE,IMG_SIZE])
    bmask_test = np.empty([0,IMG_SIZE,IMG_SIZE])
    for path in test_id:
        U0 = nib.load(os.path.join(test_folder, path, 'U0_5ch_Rig.nii')).get_fdata()
        U0 = np.squeeze(U0)

        T1mprage = nib.load(os.path.join(test_folder, path, 'T1mprage_Rig.nii')).get_fdata()
        T1gre = nib.load(os.path.join(test_folder, path, 'T1gre_Rig.nii')).get_fdata()
        T2flair = nib.load(os.path.join(test_folder, path, 'T2flair_Rig.nii')).get_fdata()
        mask = nib.load(os.path.join(test_folder, path, 'mask_auto.nii')).get_fdata()
        bmask = nib.load(os.path.join(test_folder, path, 'T1mprage_rig_brain_mask.nii.gz')).get_fdata()

        # select effective slices (slices with content)
        temp = np.sum(np.sum(mask,axis=0),axis=0)
        eff_slc = np.nonzero(temp)
        U0 = np.squeeze(U0[:,:,eff_slc,:]) # 1:-2
        T1mprage = np.squeeze(T1mprage[:,:,eff_slc])
        T1gre = np.squeeze(T1gre[:,:,eff_slc])
        T2flair = np.squeeze(T2flair[:,:,eff_slc])
        mask = np.squeeze(mask[:,:,eff_slc])
        bmask = np.squeeze(bmask[:,:,eff_slc])

        U0 = normalize_U0_slc_95prc(U0,bmask,mask)
        T1mprage = normalize_slc_95prc(T1mprage, bmask,mask)
        T1gre = normalize_slc_95prc(T1gre, bmask,mask)
        T2flair = normalize_slc_95prc(T2flair,bmask,mask)

        U0 = np.swapaxes(np.swapaxes(U0,0,2),1,2)

        input_img = U0
        X_test = np.append(X_test,input_img,0)

        output_img = np.array([T1mprage,T1gre,T2flair])
        output_img= np.swapaxes(output_img,0,3)
        Y_test =np.append(Y_test,output_img,0)

        mask_test = np.append(mask_test,np.swapaxes(np.swapaxes(mask,0,2),1,2),0)
        bmask_test = np.append(bmask_test,np.swapaxes(np.swapaxes(bmask,0,2),1,2),0)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # ----------- construct model -----------
    from Networks_sum import Unet_up
    model = Unet_up(IN_CHANNEL,OUT_CHANNEL,48,4,3,'average')
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4, beta_1=0.5))

    # ----------- load trained model -----------
    model.load_weights(checkpoint_path)

    # ----------- make predictions -----------
    test_output = model.predict(test_ds)

    # ----------- save network output as nii ----------- 
    affine = np.array([[1, 0, 0, 1],
                 [0, 1, 0, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 1]])
    empty_header = nib.Nifti1Header()
    output = nib.Nifti1Image(test_output, affine, empty_header)
    nib.save(output, result_path)

    # evaluate metrics
    n = np.shape(X_test)[0]
    nrmse_val = np.zeros([n,OUT_CHANNEL])
    psnr_val = np.zeros([n,OUT_CHANNEL])
    ssim_val = np.zeros([n,OUT_CHANNEL])
    for i in range(n):
        for j in range(OUT_CHANNEL):
            output = np.squeeze(test_output[i,:,:,j])*mask_test[i,:,:]
            label = np.squeeze(Y_test[i,:,:,j])
            nrmse_val[i,j] = normalized_root_mse(label, output)
            psnr_val[i,j] = peak_signal_noise_ratio(label, output, data_range=label.max()-label.min())
            ssim_val[i,j] = structural_similarity(label, output, data_range=label.max()-label.min())

    # evaluate within brain (after skull stripping)
    nrmse_bmask_val = np.zeros([n,OUT_CHANNEL])
    psnr_bmask_val = np.zeros([n,OUT_CHANNEL])
    ssim_bmask_val = np.zeros([n,OUT_CHANNEL])
    for i in range(n):
        for j in range(OUT_CHANNEL):
            output = np.squeeze(test_output[i,:,:,j])*bmask_test[i,:,:]
            label = np.squeeze(Y_test[i,:,:,j])*bmask_test[i,:,:]
            nrmse_bmask_val[i,j] = normalized_root_mse(label, output)
            psnr_bmask_val[i,j] = peak_signal_noise_ratio(label, output, data_range=label.max()-label.min())
            ssim_bmask_val[i,j] = structural_similarity(label, output, data_range=label.max()-label.min())

    return nrmse_val,psnr_val,ssim_val,nrmse_bmask_val,psnr_bmask_val,ssim_bmask_val


# ======= main ========
test_folder = "data_folder"
test_id = np.array(['MS_1','MS_7','MS_13'])
checkpoint_path = "checkpoint/cp-0500.ckpt"
result_path = "results/output_cv1.nii"
[nrmse_val,psnr_val,ssim_val,nrmse_bmask_val,psnr_bmask_val,ssim_bmask_val] = eval_1cvgroup(test_folder,test_id, checkpoint_path, result_path)
print("evaluation completed")

print("metrics (include skull)")
print("mean NRMSE",np.mean(nrmse_val, axis=0),"std NRMSE",np.std(nrmse_val, axis=0))
print("mean PSNR",np.mean(psnr_val, axis=0),"std PSNR",np.std(psnr_val, axis=0))
print("mean SSIM",np.mean(ssim_val, axis=0),"std SSIM",np.std(ssim_val, axis=0))
print("metrics (skull-stripped)")
print("mean NRMSE",np.mean(nrmse_bmask_val, axis=0),"std NRMSE",np.std(nrmse_bmask_val, axis=0))
print("mean PSNR",np.mean(psnr_bmask_val, axis=0),"std PSNR",np.std(psnr_bmask_val, axis=0))
print("mean SSIM",np.mean(ssim_bmask_val, axis=0),"std SSIM",np.std(ssim_bmask_val, axis=0))

