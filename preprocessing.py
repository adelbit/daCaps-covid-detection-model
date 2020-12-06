# Libraries Used
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 
INPUT_SIZE = (128,128)
mapping = {'No Findings': 0,'COVID-19': 1}

trainFile = 'train.txt'
validFile = 'validate.txt'
testFile = 'test.txt'

file = open(trainFile, 'r') 
trainfiles = file.readlines()
file = open(validFile, 'r')
validfiles = file.readlines() 
file = open(testFile, 'r')
testfiles = file.readlines()

print('No of samples for train: ', len(trainfiles))
print('No of samples for valid: ', len(validfiles))
print('No of samples for test: ', len(testfiles))

x_train = []
x_valid = []
x_test = []
y_train = []
y_valid = []
y_test = []

for i in range(len(testfiles)):
    test_i = testfiles[i].split()
    imgpath = test_i[1]
    img = cv2.imread(os.path.join(r'./data/test', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_test.append(img)
    y_test.append(mapping[test_i[2]])

print('Shape of testing images: ', x_test[0].shape)

for i in range(len(validfiles)):
    valid_i = validfiles[i].split()
    imgpath = valid_i[1]
    img = cv2.imread(os.path.join(r'./data/valid', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_valid.append(img)
    y_valid.append(mapping[valid_i[2]])

print('Shape of validation images: ', x_valid[0].shape)

for i in range(len(trainfiles)):
    train_i = trainfiles[i].split()
    imgpath = train_i[1]
    img = cv2.imread(os.path.join(r'./data/train', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_train.append(img)
    y_train.append(mapping[train_i[2]])

print('Shape of training images: ', x_train[0].shape)

# Shape of test images:  (128, 128, 3)
# Shape of train images:  (128, 128, 3)
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_valid.npy', x_valid)
np.save('y_valid.npy', y_valid)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)