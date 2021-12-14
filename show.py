import tensorflow as tf

from data_preprocessing import random_mini_batches, getTrainData

g = tf.Graph()
with g.as_default() as g:
    tf.train.import_meta_graph('./Model/model.ckpt.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='./events/', graph=g)

