# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:14:43 2022

@author: Kim Young
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

train_file_dir = r"C:\Users\Kim Young\Desktop\SHRDC\Deep Learning\TensorFlow Deep Learning\Datasets\data-science-bowl\train"
test_file_dir = r"C:\Users\Kim Young\Desktop\SHRDC\Deep Learning\TensorFlow Deep Learning\Datasets\data-science-bowl\test"

#%%
images = []
masks = []

#Load images and masks
def load_images(image_path):
    for image_file in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        images.append(img)
    return images

def load_masks(mask_path):
    for mask_file in os.listdir(mask_path):
        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128,128))
        masks.append(mask)
    return masks

train_image_dir = os.path.join(train_file_dir, 'inputs')
train_images = load_images(train_image_dir)
test_image_dir = os.path.join(test_file_dir, 'inputs')
test_images = load_images(test_image_dir)

train_mask_dir = os.path.join(train_file_dir, 'masks')
train_masks = load_masks(train_mask_dir)
test_mask_dir = os.path.join(test_file_dir, 'masks')
test_masks = load_masks(test_mask_dir)

#%%
#Convert images and masks into numpy arrays
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
#Displat examples of the images and masks    
plt.figure(figsize = (10, 4))

for i in range(1, 4):
    plt.subplot(1, 3, i)
    img_plot = images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()

plt.figure(figsize = (10, 4))

for i in range(1, 4):
    plt.subplot(1, 3, i)
    mask_plot = masks[i]
    plt.imshow(mask_plot, cmap = 'gray')
    plt.axis('off')
plt.show()

#%%
#Expand the dimensions of masks
train_masks_np_exp = np.expand_dims(train_masks_np, axis = -1)
test_masks_np_exp = np.expand_dims(test_masks_np, axis = -1)

print(train_masks[0].min(), train_masks[0].max())

#%%
#Encode into numerical encoding
converted_masks_train = np.ceil(train_masks_np_exp / 255)
converted_masks_test = np.ceil(test_masks_np_exp / 255)

converted_masks_train = 1 - converted_masks_train
converted_masks_test = 1 - converted_masks_test

#%%
#Normalize the images
converted_images_train = train_images_np / 255.0
converted_images_test = test_images_np / 255.0

#%%
#Split into train-validation split
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(converted_images_train, converted_masks_train, test_size = 0.2, random_state = 12345)

#%%
#Convert the numpy arrays into tensor slices
train_x = tf.data.Dataset.from_tensor_slices(x_train)
val_x = tf.data.Dataset.from_tensor_slices(x_val)

train_y = tf.data.Dataset.from_tensor_slices(y_train)
val_y = tf.data.Dataset.from_tensor_slices(y_val)

test_x = tf.data.Dataset.from_tensor_slices(converted_images_test)
test_y = tf.data.Dataset.from_tensor_slices(converted_masks_test)

#%%
#Zip the tensor slices into dataset
train = tf.data.Dataset.zip((train_x,train_y))
val = tf.data.Dataset.zip((val_x,val_y))
test = tf.data.Dataset.zip((test_x,test_y))

#%%
#Convert dataset into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 800/BATCH_SIZE
VALIDATION_STEP = 200/BATCH_SIZE

train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size = AUTOTUNE)
val = val.batch(BATCH_SIZE).repeat()
val = val.prefetch(buffer_size = AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size = AUTOTUNE)

#%%
from tensorflow_examples.models.pix2pix import pix2pix

#Define the model
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top = False)

# Use activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
 
  #Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])
  
  #Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 2
model = unet_model(output_channels = OUTPUT_CLASSES)

#%%
model.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
model.summary()

#%%
#Function to display sample images and masks
def display(display_list):
    plt.figure(figsize = (15, 15))
    title = ['Input Imgae', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
for images, masks in train.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

#%%
import datetime
from IPython.display import clear_output

#Funtion to process predicted mask
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis = -1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

#Function to show predictions
def show_prediction(dataset = None, num = 1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)[0]])
            
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))[0]])
  
#Function to observe how model improve while training
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        clear_output(wait = True)
        show_prediction()
        print('\n Sample predictions after epoch {} \n'.format(epoch + 1))
        
#%%
log_dir = r'C:\Users\Kim Young\Desktop\SHRDC\Deep Learning\TensorFlow Deep Learning\Tensorboard\logs'
tb_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq = 1, profile_batch = 0)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)

#Perform training
EPOCH = 100

history = model.fit(train, epochs = EPOCH, 
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    validation_steps = VALIDATION_STEP, 
                    validation_data = test, 
                    callbacks = [DisplayCallback(), tb_callback, es_callback])

#%%
#Test evaluation
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss = {test_loss}")
print(f"Test accuracy = {test_accuracy}")

#%%
#Make some predictions
show_prediction(test, 3)
