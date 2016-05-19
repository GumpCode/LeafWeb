#!/usr/bin/python
# -*- coding: UTF-8 -*-

#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
from pylab import *
from PIL import Image
import MySQLdb
import os
import cv2


#种类名字
plantNames=['Populus alba', 'Quercus suber', 'Salix atrocinerea', 'Populus nigra', 'Alnus sp', 'Quercus robur', 'Crataegus monogyna', 'Ilex aquifolium', 'Nerium oleander', 'Betula pubescens', 'Tilia tomentosa', 'Acer palmaturu', 'Celtis sp', 'Corylus avellana', 'Castanea sativa']
#图片保存目录
leaf_dir = '/usr/local/tomcat/webapps/plant/photos'
#图片大小
w=64
h=64


#print 'init a model'
#init a model  根据已生成的结构和参数文件初始化两个网络,一个用于判断是否是叶子，一个用于判断叶子种类
model1 = Sequential()
model1 = model_from_json(open('isLeaf.json').read())
model1.load_weights('isLeaf_weight.h5')
model2 = Sequential()
model2 = model_from_json(open('model_architecture.json').read())
model2.load_weights('model_weight.h5')
#print 'build network completed'


print 'start predicting'
try:
    conn=MySQLdb.connect(host='localhost', user='root', passwd='root', db='bookshop', charset='utf8')
    cur=conn.cursor()
    cur.execute('select pictureURL from picture where recognizedBy is null')  #根据recognizedBy选出没有处理的图片
    result=cur.fetchall()
    count = 0
    if result:
        for item in result:
            predict_data=[]
            image=str(item).strip('(\'').strip('\',)').strip('u\'')
            path=os.path.join(leaf_dir, image)
            #判断需要处理的图片是否存在
            if not os.path.exists(path):
                continue
            img=cv2.imread(path)
            im=array(cv2.resize(img,(w, h)))
            predict_data.append([im[:,:,0], im[:,:,1], im[:,:,2]])
            predict_data=asarray(predict_data, float32)
            #判断是否是叶子
            classes_result=model1.predict_classes(predict_data, batch_size=1, verbose=0)
            pro_result=model2.predict_proba(predict_data, batch_size=1, verbose=0)
            cur.execute('update picture set recognizedBy=\'machine\' where pictureURL=\'' + image + '\'')
            if classes_result[0] == 1: #如果是叶子，继而判断叶子种类
                classes_result=model2.predict_classes(predict_data, batch_size=1, verbose=0)
                pro_result=model2.predict_proba(predict_data, batch_size=1, verbose=0)
                accuracy = round(float(pro_result[:,int(classes_result[0])]), 2)
                plantName=plantNames[int(classes_result[0])]
                if accuracy >= 0.5: #如果识别率达到0.5，返回种类名字、识别率
                    cur.execute('update picture set plantName=\'' + plantName  + '\' where pictureURL=\'' + image + '\'')
                    cur.execute('update picture set accuracyRate=\'' + str(accuracy) + '\' where pictureURL=\'' + image + '\'')
                    #cur.execute('update picture set recognizedBy=\'machine\' where pictureURL=\'' + image + '\'')
            else:#如果不是叶子，修改plantName为no
                cur.execute('update picture set plantName=\'no\' where pictureURL=\'' + image + '\'')
#               cur.execute('update picture set accuracyRate=null' where pictureURL=\'' + image + '\'')
            count = count + 1

    conn.commit()
    cur.close()
    conn.close()
    print str(count) + ' images are processed'

except MySQLdb.Error, e:
    print "Mysql Error %d: %s" % (e.args[0], e.args[1])
