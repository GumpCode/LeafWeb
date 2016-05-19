#coding: utf-8

#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from pylab import *
from PIL import Image
import numpy as np
import os
import cv2


batch_size =30  #迭代中批大小
nb_epoch = 100  #迭代次数
data_augmentation = True
img_rows, img_cols = 64,64  #图片长宽
img_channels = 3  #图片通道，3代表彩色
nb_classes = 2 #种类个数
classes=range(0,2)

#读入数据
def load_data(dirName):
    data=[]
    label=[]
    for i in os.listdir(dirName):
        im=array(Image.open(dirName+"/"+i))
        l=int(i.split("_")[0])
        if l in classes:
            data.append([im[:,:,0],im[:,:,1],im[:,:,2]])
            label.append([classes.index(l)])
    data=asarray(data,float32)
    label=asarray(label,uint8)
    return data,label
training_data,training_label=load_data("Flip")
training_label = np_utils.to_categorical(training_label, nb_classes)
print ('finish loading data')


#init a model
model = Sequential()

#conv1 卷积层
model.add(Convolution2D(32, 3, 3,  input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('tanh')) #激活函数
model.add(MaxPooling2D(pool_size=(2, 2))) #下采样
model.add(Dropout(0.25)) #避免过拟合

#conv2 卷积层
model.add(Convolution2D(32, 4,4))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2))) #下采样
model.add(Dropout(0.5))

#conv3
model.add(Convolution2D(64, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

#conv4
model.add(Convolution2D(64, 5, 5))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#展开，得到结果
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam")
result = model.fit(training_data, training_label, batch_size=batch_size,nb_epoch=nb_epoch,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.7)

#保存网络结构和参数
json_string = model.to_json()
open('model_isLeaf.json', 'w').write(json_string)
model.save_weights('isLeaf_weight.h5')
