#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
from glob import glob


# In[2]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision


# In[3]:


from model import build_unet
from metrics import dice_coef, iou


# In[4]:


#defining height and width 
H = 256
W = 256


# In[5]:


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[6]:


#x refers to the image data while Y refers to the masks
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


# In[7]:


def load_data(dataset_path,split=0.2):
    #data here is images plus masks
    #use glob function since you are pointing to the extension 
    
    images=sorted(glob(os.path.join(dataset_path, "ISBI2016_ISIC_Part1_Training_Data", "*.jpg")))
    masks=sorted(glob(os.path.join(dataset_path, "ISBI2016_ISIC_Part1_Training_GroundTruth", "*.png")))
    
    #calculate the test size 
    test_size = int(len(images) * split)
    #creating training/validation images/masks splitting 
    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)
    #creating test splitting
    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
    
    return (train_x, train_y), (valid_x, valid_y),(test_x,test_y)
    


# In[8]:


#creating input pipeline 
#function to read images -> takes path, read image as RGB,resize it to the defined HW, normalize it
#size (256,256,3)
#function to read masks -> takes path , read mask as GrayScale,resize it to the defined HW, normalize it
#size should be (256,256,1)
#expand dimension to add the gray scale channel 
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x 

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)  ## (256, 256)
    #expanding dimensions to add the extra gray scale channel
    x = np.expand_dims(x, axis=-1)              ## (256, 256, 1)
    return x


# In[9]:


#parsing converting the reading functions into tensorflow format
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


# In[10]:


def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


# In[15]:


if __name__ == "__main__":
    """ environment needs to be seeded """
    np.random.seed(42)
    tf.random.set_seed(42)
    """creating a dictory to save the data """
    create_dir("files")
    """creating hyper parameters"""
    batch_size = 4
    lr = 1e-4 ## (0.0001)
    num_epoch = 15
    model_path = "files/model"
    #csv file for saving all the metrics during training
    csv_path = "files/data.csv"
    """dataset loader"""
    dataset_path = "training_data/"
    
    (train_x, train_y), (valid_x, valid_y),(test_x,test_y) = load_data(dataset_path)
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)
    #calculating steps which is the number of batches
    #training steps =769/4=192
    #validation steps = 255/4= 63
    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size
    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1
    
    
    "model building"
    #build_unit method takes input shape which is hxwx3
    model = build_unet((H, W, 3))
    #metrics to evaluate our training are dice coefficient,iou,recall and precision
    #Precision quantifies the number of positive class predictions that actually belong to the positive class
    #truePositive/totalPredictedPositives
    #predictedPositive=TruePositive+falsePositive
    #Recall quantifies the number of positive class predictions made out of all positive examples in the dataset
    #TruePositive/TotalActualPositive
    #actualPositive=FalseNegative+truePositive
    metrics = [dice_coef, iou, Recall(), Precision()]
    
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    model.summary()
    #model check point saves model weights during training
    #reduce learning rate on plateau reduces learning rate by factor of 0.1 after no validation loss enhancement 
    #over 5 epochs
    #CSVlogger to store the logs of the training in csv file
    #tensorboard to visualize the paramters during training
    #earlyStopping to stop training when validation loss doesnt improve for 20 epochs 
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]
    
    
    model.fit(
        train_dataset,
        epochs=num_epoch,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )
  
    


# In[ ]:




