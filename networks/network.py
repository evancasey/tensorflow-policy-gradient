import tensorflow as tf
import numpy as np

num_features = 2

trX = np.array([[1,2],[0,1]])
trY = np.array([[1],[0]])

X = tf.placeholder(tf.float32, [None, num_features])

w = tf.Variable(tf.random_normal(shape=[num_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1]))

model = tf.matmul(X, w)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            x = sess.run(model, feed_dict={X: x})
            import ipdb; ipdb.set_trace()

