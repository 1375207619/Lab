import math
import tensorflow as tf
import pandas as pd
import numpy as np


#转化为one-hot编码,方便训练
def toOneHot(array):
    return np.eye(10)[array.reshape(-1)]

#获得训练数据
def getTrainData():
    train = pd.read_csv('data/train.csv')
    X_train = train.iloc[:40000, 1:].values
    # 将输入值区间调整到[0,1]
    X_train = np.float32(X_train / 255.)
    # 将行向量转换为图片
    X_train = np.reshape(X_train, [40000, 28, 28, 1])
    Y_train = train.iloc[:40000, 0].values
    return  X_train,toOneHot(Y_train)

#获得测试数据
def getDevData():
    train = pd.read_csv('data/train.csv')
    X_dev = train.iloc[40000:, 1:].values
    X_dev = np.float32(X_dev / 255.)
    X_dev = np.reshape(X_dev, [2000, 28, 28, 1])
    Y_dev = train.iloc[40000:, 0].values
    return X_dev,toOneHot(Y_dev)



#随机抽取batch个数据,然后制作成mini_batch,提升训练速度
def random_mini_batches(X, Y, mini_batch_size):
    m = X.shape[0]
    mini_batches = []

    # 打乱数据
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # 制作mini batch
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 如果样本数不能被mini batch的size整除，则将剩下的整合起来
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False


X_train,Y_train=getTrainData()
for i in range(5):
    img = np.reshape(X_train[i, :],[28,28])
    label = np.argmax(Y_train[i, :])
    plt.matshow(img, cmap = plt.get_cmap('gray'))
    plt.title('第%d张图片 标签为%d' %(i+1,label))
    plt.show()