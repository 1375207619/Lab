import math

import numpy as np
import pandas as pd
import  tensorflow as tf
import os
import  matplotlib.pyplot as plt

from data_preprocessing import getTrainData, getDevData, random_mini_batches

gpu_no = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
# 定义TensorFlow配置
config = tf.ConfigProto()
# 配置GPU内存分配方式，按需增长
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)




#设置3个placeholder
with tf.name_scope('input'):
    #输入层
    X = tf.placeholder(tf.float32, (None, 28, 28, 1), name='X')
    Y = tf.placeholder(tf.float32, (None, 10))

#这个placeholder是为了在训练的时候开启dropout，测试的时候关闭dropout
training = tf.placeholder(tf.bool, name = 'training')


def weight_variable(shape):
    # tf.truncated_normal从截断的正态分布中输出随机值.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #初始化偏置项,设置为非零免死神经元
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # 采用最大池化，也就是取窗口中的最大值作为结果
    # x 是一个4维张量，shape为[batch,height,width,channels]
    # ksize表示pool窗口大小为2x2,也就是高2，宽2
    # strides，表示在height和width维度上的步长都为2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(X, [-1, 28, 28, 1])

##conv1 包含pool
# 初始化W_conv0为[3,3,1,32]的张量tensor，表示卷积核大小为3*3，1表示图像通道数（输入），32表示卷积核个数即输出32个特征图（即下一层的输入通道数）
with tf.name_scope('Conv1'):
    with tf.name_scope('W_conv1'):
        W1 = tf.get_variable('W1', [3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32])
    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W1) + b_conv1)  # output size 28x28x32 3x3x1的卷积核作用在28x28x1的二维图上
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32 卷积操作使用padding保持维度不变，只靠pool降维


## conv2 layer 含pool##
with tf.name_scope('Conv2'):
    with tf.name_scope('W_conv2'):
        W2 = weight_variable([3, 3, 32, 64])  # 同conv1，不过卷积核数增为64
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64])
    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b_conv2)  # output size 14x14x64
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

## 全连接层1 ##
with tf.name_scope('Fc1'):
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024])
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('h_fc1_drop'):
        dropout_3 = tf.contrib.layers.dropout(h_fc1, keep_prob=0.5, is_training=training)

## 全连接层2 ##
# 含10个神经元
with tf.name_scope('Fc2'):
    with tf.name_scope('fc2'):
        full_2 = tf.contrib.layers.fully_connected(dropout_3, 10, activation_fn=None)
    with tf.name_scope('pediction'):
        prediction = tf.argmax(full_2, 1, name='prediction')




with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=full_2, labels=Y))#交叉熵代价函数和softmax搭配
    tf.summary.scalar('cost',cost)



starter_learning_rate = 1e-4
global_step = tf.Variable(0, trainable=False)
#当快接近全局最优点时，如果学习率很大会在最优点附近震荡，所以随之迭代的次数增多需要减小学习率
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1500, 0.96, staircase = True)
#使用梯度下降法
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#获得训练集
X_train,Y_train=getTrainData()
#获得测试集
X_dev,Y_dev=getDevData()
num_epochs = 1100
#batch
minibatch_size = 100
costs = []
m = X_train.shape[0]

with tf.Session() as sess:
    sess.run(init)#初始化参数
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size)
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y,training:1})
            minibatch_cost += temp_cost / num_minibatches
        if epoch %20  == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            correct_prediction = tf.equal(tf.argmax(full_2,1), tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Train Accuracy:", accuracy.eval({X: X_train[:4000,:,:,:], Y: Y_train[:4000,:], training:0}))
            Dev_Accuracy = accuracy.eval({X: X_dev, Y: Y_dev, training:0})
            print ("Dev Accuracy:", Dev_Accuracy)
            print(sess.run(global_step))
            print(sess.run(learning_rate))#为了观察学习率
            if Dev_Accuracy >=0.992 or epoch == 1000:
                print('训练完成')
                saver.save(sess,'./Model/model.ckpt')#保存网络
                break
        if epoch % 1 == 0:
            costs.append(minibatch_cost)#记录cost

