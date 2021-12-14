import tensorflow as tf
import pandas as pd
import numpy as np
test = pd.read_csv('data/test.csv')

X_test = test.iloc[:,:].values
X_test = np.float32(X_test/255.)
X_test = np.reshape(X_test, [28000,28,28,1])

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./Model/model.ckpt.meta')
    saver.restore(sess, "./Model/model.ckpt")
    graph = tf.get_default_graph()
    graph = tf.get_default_graph()

    graph_op = graph.get_operations()
    for i in graph_op:
        print(i)
    X = graph.get_tensor_by_name("input/X:0")
    training = graph.get_tensor_by_name("training:0")
    prediction = graph.get_tensor_by_name("Fc2/pediction/prediction:0")
    for i in range(7):
        prediect = sess.run(prediction, feed_dict={X:X_test[i*4000:(i+1)*4000,:,:,:], training:0})
        try:
            result = np.hstack((result,prediect))
        except:
            result = prediect
    pd.DataFrame({"ImageId": range(1, len(result) + 1), "Label": result}).to_csv('data\submission.csv', index=False, header=True)