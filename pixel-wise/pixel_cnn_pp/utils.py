import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors import FailedPreconditionError

def make_feed_data(data, bitlen):
    if type(data) is tuple:
        x, y = data
    else:
        x = data
        y = None
    
    x = num_to_bit(x, bitlen)
    x = np.transpose(x, [0, 1, 2, 4, 3])
    shape = x.shape
    x = np.reshape(x, shape[:3] + (np.prod(shape[-2:]),))
    return x, y
    

def scale_function(x):
    return np.cast[np.float32]((x-127.5) / 127.5)  

def num_to_bit(x, bitlen):
    BITLEN = bitlen
    assert x.ndim == 4
    n, h, w, c = x.shape
    x_out = np.zeros([n, h, w, c, BITLEN])
    for i_n in range(n):
        for i_h in range(h):
            for i_w in range(w):
                for i_c in range(c):
                    string = bin(x[i_n, i_h, i_w, i_c])[2:]
                    binary = [int(x) for x in string]  
                    bits = [0] * ( 8 - len(binary))
                    bits.extend(binary)
                    assert len(bits) == 8
                    x_out[i_n, i_h, i_w, i_c, :] = bits[:BITLEN]
    return x_out
    
    
def bit_to_num(x, bit_len):
    assert x.ndim == 5
    n, h, w, c, bitlen = x.shape
    x_out = np.zeros([n, h, w, c])
    for i_n in range(n):
        for i_h in range(h):
            for i_w in range(w):
                for i_c in range(c):
                    bits = [str(int(bit)) for bit in x[i_n, i_h, i_w, i_c, :]]
                    bits.extend(['0'] * (8 - bit_len))
                    bit_string = ''.join(bits)
                    assert len(bits) == 8
                    num = int(bit_string, 2)
                    
                    x_out[i_n, i_h, i_w, i_c] = num
    return x_out 
    
    
def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]), feed_dict)
            except FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happend if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or external precondition unsatisfied.")
        else:
            vars_left = new_vars_left
    