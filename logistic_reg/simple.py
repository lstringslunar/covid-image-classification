import pandas as pd
import scipy
from sklearn import datasets, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from random import sample
import random
import numpy

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
train_data = [['train/' + f, int(k == 'positive')] for f, k in zip(train_df['filename'], train_df['class'])]
# train_data = [['test/' + f, int(k == 'positive')] for f, k in zip(test_df['filename'], test_df['class'])]  # <<<
test_data = [['test/' + f, int(k == 'positive')] for f, k in zip(test_df['filename'], test_df['class'])]
# --- ---------------------------------------------------------------------------------------- ---
# preparing datasets
print('preparing training set...')
size = (16, 16)
sample_size = 20000
random.seed(12345)
sample_train = sample(train_data, sample_size)
# sample_train = train_data  # <<<
train_set = []
for f in tqdm(sample_train):
    path = f[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size, cv2.INTER_AREA)
    img = img.flatten()
    train_set.append(img)
train_label = [row[1] for row in sample_train]
# --- --------------------------------------- ---
# plt.subplot(1, 2, 1)  # row 1, col 2 index 1
# plt.imshow(train_set[0], cmap='gray')
# plt.title("First Image")
# plt.xlabel('Negative')
#
# plt.subplot(1, 2, 2)  # index 2
# plt.imshow(train_set[1], cmap='gray')
# plt.title("Second Image")
# plt.xlabel('Positive')
# plt.savefig('logistic_reg_sample.png')
# --- --------------------------------------- ---
# scaler = preprocessing.StandardScaler().fit(train_set)
# train_set = scaler.transform(train_set)

print('preparing test set...')
test_set = []
for t in tqdm(test_data):
    path = t[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size, cv2.INTER_AREA)
    img = img.reshape(1, -1)
    test_set.append(img)
"""
test_set_ori = []
for t in tqdm(test_data):
    path = t[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size, cv2.INTER_AREA)
    img = img.flatten()
    test_set_ori.append(img)
scaler = preprocessing.StandardScaler().fit(test_set_ori)
test_set_ori = scaler.transform(test_set_ori)
test_set = []
for img in test_set_ori:
    img = img.reshape(1, -1)
    test_set.append(img)
"""
test_label = [row[1] for row in sample_train]
# --- ---------------------------------------------------------------------------------------- ---
# training (Logistic Regression)
model = LogisticRegression(solver='lbfgs', max_iter=3000)
model.fit(train_set, train_label)

tp, tn, fp, fn = 0, 0, 0, 0
for img, label in zip(test_set, test_label):
    pred = model.predict(img)
    if pred == label:
        if pred == 0:
            tn += 1
        else:
            tp += 1
    else:
        if pred == 0:
            fn += 1
        else:
            fp += 1

print(tp, tn, fp, fn)

accuracy = (tp + tn) / (tp + fp + tn + fn)

print(f'Accuracy = {accuracy:0.3f}')
