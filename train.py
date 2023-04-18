# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 2020
@author: Shihan Qiu (qiushihan19@gmail.com)
"""

import os

# select GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

import numpy as np
import nibabel as nib
import tensorflow as tf

# parameters to set
IMG_SIZE = 256
CROP_H = 160  # patch size
CROP_W = 128

IN_CHANNEL = 5
OUT_CHANNEL = 3

BATCH_SIZE = 4
BUFFER_SIZE = 500  # buffer size for shuffle
N_EPOCHS = 500
CKPT_PERIOD = 500


# ----------- data augmentation ----------- 
def augment(input_img,output_img):
    image = tf.concat([input_img,output_img],2)
    #image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE+60, IMG_SIZE+60) # Add padding
    image = tf.image.random_crop(image, size=[CROP_H, CROP_W, IN_CHANNEL+OUT_CHANNEL]) # Random crop back
    image = tf.image.random_flip_left_right(image) # Random flip left right
    image = tf.image.random_flip_up_down(image) # Random flip up down
    input_img = image[:,:,0:IN_CHANNEL]
    output_img = image[:,:,IN_CHANNEL:IN_CHANNEL+OUT_CHANNEL]
    return input_img,output_img

# ----------- loss function ----------- 
def SSIM_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, tf.math.reduce_max(y_true)-tf.math.reduce_min(y_true)))

def L1_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.abs(y_true - y_pred))

def Mixed_loss(y_true, y_pred):
    return L1_loss(y_true, y_pred) + 0.1*SSIM_loss(y_true, y_pred)

# ----------- training for one cv fold ----------- 
from normalization import normalize_U0_slc_95prc, normalize_slc_95prc

def train_1cvgroup(data_folder,train_id, test_id, checkpoint_path):
    # load data and construct dataset
    print("load the data ...")
    print("train data:",train_id)
    print("test data:",test_id)

    X_train = np.empty([0,IMG_SIZE,IMG_SIZE,IN_CHANNEL])
    Y_train = np.empty([0,IMG_SIZE,IMG_SIZE,OUT_CHANNEL])
    for path in train_id:
        U0 = nib.load(os.path.join(data_folder, path, 'U0_5ch_Rig.nii')).get_fdata()
        U0 = np.squeeze(U0)
        T1mprage = nib.load(os.path.join(data_folder, path, 'T1mprage_Rig.nii')).get_fdata()
        T1gre = nib.load(os.path.join(data_folder, path, 'T1gre_Rig.nii')).get_fdata()
        T2flair = nib.load(os.path.join(data_folder, path, 'T2flair_Rig.nii')).get_fdata()
        mask = nib.load(os.path.join(data_folder, path, 'mask_auto.nii')).get_fdata()
        bmask = nib.load(os.path.join(data_folder, path, 'T1mprage_rig_brain_mask.nii.gz')).get_fdata()

        # select effective slices (slices with content)
        temp = np.sum(np.sum(mask,axis=0),axis=0)
        eff_slc = np.nonzero(temp)
        U0 = np.squeeze(U0[:,:,eff_slc,:]) # 1:-2
        T1mprage = np.squeeze(T1mprage[:,:,eff_slc])
        T1gre = T1gre[:,:,eff_slc]
        T2flair = np.squeeze(T2flair[:,:,eff_slc])
        mask = np.squeeze(mask[:,:,eff_slc])
        bmask = np.squeeze(bmask[:,:,eff_slc])

        U0 = normalize_U0_slc_95prc(U0,bmask,mask)
        T1mprage = normalize_slc_95prc(T1mprage,bmask,mask)
        T1gre = normalize_slc_95prc(T1gre, bmask,mask)
        T2flair = normalize_slc_95prc(T2flair,bmask,mask)

        input_img = U0
        input_img = np.swapaxes(np.swapaxes(input_img,0,2),1,2)
        X_train = np.append(X_train,input_img,0)
        
        output_img = np.array([T1mprage,T1gre,T2flair])
        output_img = np.swapaxes(output_img,0,3)
        Y_train = np.append(Y_train,output_img,0)
        
    X_test = np.empty([0,IMG_SIZE,IMG_SIZE,IN_CHANNEL])
    Y_test = np.empty([0,IMG_SIZE,IMG_SIZE,OUT_CHANNEL])
    for path in test_id:
        U0 = nib.load(os.path.join(data_folder, path, 'U0_5ch_Rig.nii')).get_fdata()
        U0 = np.squeeze(U0)
        T1mprage = nib.load(os.path.join(data_folder, path, 'T1mprage_Rig.nii')).get_fdata()
        T1gre = nib.load(os.path.join(data_folder, path, 'T1gre_Rig.nii')).get_fdata()
        T2flair = nib.load(os.path.join(data_folder, path, 'T2flair_Rig.nii')).get_fdata()
        mask = nib.load(os.path.join(data_folder, path, 'mask_auto.nii')).get_fdata()
        bmask = nib.load(os.path.join(data_folder, path, 'T1mprage_rig_brain_mask.nii.gz')).get_fdata()

        temp = np.sum(np.sum(mask,axis=0),axis=0)
        eff_slc = np.nonzero(temp)
        U0 = np.squeeze(U0[:,:,eff_slc,:]) # 1:-2
        T1mprage = np.squeeze(T1mprage[:,:,eff_slc])
        T1gre = T1gre[:,:,eff_slc]
        T2flair = np.squeeze(T2flair[:,:,eff_slc])
        mask = np.squeeze(mask[:,:,eff_slc])
        bmask = np.squeeze(bmask[:,:,eff_slc])

        U0 = normalize_U0_slc_95prc(U0,bmask,mask)
        T1mprage = normalize_slc_95prc(T1mprage,bmask,mask)
        T1gre = normalize_slc_95prc(T1gre,bmask,mask)
        T2flair = normalize_slc_95prc(T2flair,bmask,mask)

        input_img = U0
        input_img = np.swapaxes(np.swapaxes(input_img,0,2),1,2)
        X_test = np.append(X_test,input_img,0)
        
        output_img = np.array([T1mprage,T1gre,T2flair])
        output_img = np.swapaxes(output_img,0,3)
        Y_test = np.append(Y_test,output_img,0)
        

    # construct dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_ds = (
        train_ds
        .map(augment)
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    print("size of training dataset",np.shape(X_train))
    print("size of testing dataset",np.shape(X_test))

    # number of steps per epoch
    N_STEPS = np.ceil(np.shape(X_train)[0]/BATCH_SIZE)

    # ----------- construct model -----------
    from Networks_sum import Unet_up
    import datetime

    model = Unet_up(IN_CHANNEL,OUT_CHANNEL,48,4,3,'average')

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4, beta_1=0.5),
        loss = Mixed_loss)

    # ----------- train model -----------
    # tensorboard callbck
    logdir = os.path.join("log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # save checkpoints
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=CKPT_PERIOD)

    # adaptive learning rate
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,patience=10, min_lr=0)
    def scheduler(epoch, lr):
        if epoch < 300:
            return lr
        else:
            return lr * tf.math.exp(-0.02)
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(train_ds,epochs=N_EPOCHS,steps_per_epoch=N_STEPS,validation_data=test_ds,
    		    callbacks = [tensorboard_callback,cp_callback,reduce_lr])

    print("completed")
    print("size of training dataset",np.shape(X_train))
    print("size of testing dataset",np.shape(X_test))


# ======= main ========
data_folder = "data_folder"
train_id = np.array(['MS_2','MS_8','MS_14','MS_3','MS_9','MS_15','MS_4','MS_10','H_1','MS_5','MS_11','H_2','MS_6','MS_12','H_3'])
test_id = np.array(['MS_1','MS_7','MS_13'])

checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"

train_1cvgroup(data_folder, train_id, test_id, checkpoint_path)
print("training completed")
