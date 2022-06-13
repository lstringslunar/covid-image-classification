# Code Source and Explaination by Sohom Majumder
# Site : https://www.kaggle.com/code/sohommajumder21/resnet-covidx-beginner-friendly-codes-explained

import numpy as np
import pandas as pd
import os
import PIL
import cv2
import tensorflow as tf
from tensorflow import keras
# from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import shutil
from sklearn.metrics import confusion_matrix, classification_report

epochs = int(input('enter epochs: '))
if epochs <= 0:
    epochs = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# --- ---------------------------------------------------------------------------------------- ---
# Read data (Panda Dataframe)
train_df = pd.read_csv('train.txt', sep=' ', header=None)
train_df.columns = ['id', 'filename', 'class', 'source']
train_df = train_df.drop(['id', 'source'], axis=1)
print(train_df.head())
test_df = pd.read_csv('test.txt', sep=' ', header=None)
test_df.columns = ['id', 'filename', 'class', 'source']
test_df = test_df.drop(['id', 'source'], axis=1)
print(test_df.head())

# Directory
train_path = 'train/'
test_path = 'test/'

# --- ---------------------------------------------------------------------------------------- ---
# Resample
# sample_count = 5000
# print('Before resampling:')
# print(train_df['class'].value_counts())

# negative = train_df[train_df['class'] == 'negative']  # negative values in class column
# positive = train_df[train_df['class'] == 'positive']  # positive values in class column
# from sklearn.utils import resample
#
# negative_resampled = resample(negative, replace=True, n_samples=sample_count)
# positive_resampled = resample(positive, replace=True, n_samples=sample_count)
#
# train_df = pd.concat([positive_resampled, negative_resampled])

from sklearn.utils import shuffle

train_df = shuffle(train_df)

# print('After resampling:')
# print(train_df['class'].value_counts())

# --- ---------------------------------------------------------------------------------------- ---
# Validation Set
train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=0)

# --- ---------------------------------------------------------------------------------------- ---
# Let's see how many images for training and validation and testing

print(f"Training set:\n{train_df['class'].value_counts()}")
print(f"Validation set:\n{valid_df['class'].value_counts()}")
print(f"Test set:\n{test_df['class'].value_counts()}")

# --- ---------------------------------------------------------------------------------------- ---
# Generate train/valid/test image data (ImageDataGenerator)
train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

# Now fit the them to get the images from directory (name of the images are given in dataframe) with augmentation

target_size = (300, 300)
batch_size = 64
train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_path, x_col='filename',
                                              y_col='class', target_size=target_size, batch_size=batch_size,
                                              class_mode='binary')
valid_gen = test_datagen.flow_from_dataframe(dataframe=valid_df, directory=train_path, x_col='filename',
                                             y_col='class', target_size=target_size, batch_size=batch_size,
                                             class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(dataframe=test_df, directory=test_path, x_col='filename',
                                            y_col='class', target_size=target_size, batch_size=batch_size,
                                            class_mode='binary')
# class mode binary because we want the classifier to predict covid or not

# --- ---------------------------------------------------------------------------------------- ---

import tensorflow as tf

# Our base model is InceptionResNetV2, new readers are encouraged to see the architecture of this particular model

base_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=(target_size[0], target_size[1], 3),
                                              include_top=False)

for layer in base_model.layers:
    layer.trainable = False
# --- ---------------------------------------------------------------------------------------- ---

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("covid_classifier_model.h5", save_best_only=True, verbose=0),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- ---------------------------------------------------------------------------------------- ---
print('\n\nTrain: ')
history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=[callbacks])

# --- ---------------------------------------------------------------------------------------- ---
print('\n\nTest: ')
model.load_weights('./covid_classifier_model.h5')
model.evaluate(test_gen)

# --- ---------------------------------------------------------------------------------------- ---
# preds = (model.predict(test_gen) > 0.5).astype("int32")
# print(preds)
# print(history.history)
#
# --- ---------------------------------------------------------------------------------------- ---
# Train Result
from datetime import datetime

now = datetime.now().strftime("%m%d-%H%M%S")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
ep = range(1, len(acc) + 1)
plt.plot(ep, acc, 'k', label='Training acc')
plt.plot(ep, val_acc, 'b', label='Validation acc')
plt.title('Training and validation Accuracy')
plt.legend(loc='upper right')
plt.savefig('accuracy-' + now + '.png')
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(ep, loss, 'k', label='Training loss')
plt.plot(ep, val_loss, 'b', label='Validation loss')
plt.title('Training and validation Loss')
plt.legend(loc='upper right')
plt.savefig('loss-' + now + '.png')

plt.show()

file = open('model-' + now + '.txt', 'w')
text = ''
text += str((epochs, sample_count, target_size, batch_size))
file.write(text)
file.close()
# --- ---------------------------------------------------------------------------------------- ---
