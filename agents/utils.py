import scipy 
import scipy.signal
import tensorflow as tf

def discount_cum_sum(x, discount):
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]

def standardize(tensor):
    mean = tf.reduce_mean(tensor)
    variance = tf.reduce_mean(tf.square(tensor - mean))
    std_dev = tf.sqrt(variance)
    return (tensor - mean) / std_dev
