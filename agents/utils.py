import scipy 
import scipy.signal

def discount_cum_sum(x, discount):
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]
