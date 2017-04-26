import pixel_cnn_pp.nn as nn
import numpy as np
import pixel_cnn_pp.utils as U
import tensorflow as tf
from pixel_cnn_pp.model import model_spec

class PxCNN_bw(object):
    
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, class_conditional, dim, channel, bit_len,
                 batch_size, batch_size_init, dropout_p, nr_gpu, num_labels,
                 polyak_decay):
        self.nr_resnet = nr_resnet
        self.nr_filters = nr_filters
        self.resnet_nonlinearity = resnet_nonlinearity
        self.class_conditional = class_conditional
        self.batch_size = batch_size
        self.batch_size_init = batch_size_init
        self.dropout_p = dropout_p
        self.nr_gpu = nr_gpu
        self.dim = dim
        self.bit_len = bit_len
        self.num_labels = num_labels
        self.channel = channel
        self.polyak_decay = polyak_decay
        self._init()
        
    def _init(self):
        self.x_init = tf.placeholder(tf.float32, shape=(self.batch_size_init,) + (self.dim, self.dim, self.channel * self.bit_len))
        self.x      = tf.placeholder(tf.float32, shape=(self.batch_size,)      + (self.dim, self.dim, self.channel * self.bit_len))
        
        if self.class_conditional:
            self.y_init = tf.placeholder(tf.int32, shape=(self.batch_size_init,))
            self.h_init = tf.one_hot(self.y_init, depth=self.num_labels)
            
            self.y = tf.placeholder(tf.int32, shape=(self.batch_size,))
            self.h = tf.one_hot(self.y, depth=self.num_labels)
        else:
            self.y_init = None
            self.h_init = None
            self.y = None
            self.h = None
            
        # keep track of moving average
        all_params = tf.trainable_variables()
        self.ema = tf.train.ExponentialMovingAverage(decay=self.polyak_decay)
        self.maintain_averages_op = tf.group(self.ema.apply(all_params))

        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        self.loss_and_grad(self.x_init, self.h_init, init=True, reuse=False, ema=None)
        self.initializer = tf.global_variables_initializer()    
        
        self.total_loss, self.total_optimizer = self.loss_and_grad(self.x, self.h, init=False, reuse=True, ema=None)
        
        adam_vars = [v for v in tf.all_variables() if 'Adam' in v.name or 'beta' in v.name]
        print('found %d adam1 vars' %len(adam_vars))
        self.adam_init = tf.variables_initializer(adam_vars)  
        self.adam_vars = adam_vars
        
        self.inference()
        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(config=config)
        
        self.saver = tf.train.Saver()
            
    def loss_and_grad(self, x, y=None, init=False, reuse=False, ema=None):
        self.model_opt = {'nr_resnet': self.nr_resnet, 'nr_filters': self.nr_filters,
             'resnet_nonlinearity': self.resnet_nonlinearity}

        cd = [None] * self.nr_gpu
        losses = []
        optimizers = []
        gpu = 0
        
        for i in range(self.bit_len):
            for c in range(self.channel):
                with tf.device('/gpu:%d' % gpu):
                    with tf.control_dependencies(cd[gpu]):
                        print('bit_len at %d color channel at %d' %(i, c))
                        name = 'model_bit_' + str(i) + '_color' + str(c)
                        
                        x_in   = tf.expand_dims(x[:, :, :, self.channel*i+ c], 3)
                        if self.channel*i + c>0:
                            x_cond = x[:, :, :, :(self.channel*i + c)]
                            x_input = tf.concat([x_cond, x_in], 3)
                        else:
                            x_cond = None
                            x_input = x_in
                            
                        gen_par = model_spec(x_input, h=y, cond=x_cond, init=init, reuse=not init, ema=None, 
                                             dropout_p=self.dropout_p, scope=name, **self.model_opt)
                        
                        cd[gpu] = [gen_par]
                        
                        """Bernoulli loss"""
                        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_par, labels=x_in))
                        losses.append(loss)
                        
                        if not init:
                            params = [v for v in tf.trainable_variables() if name in v.name ]
                            optimizer = tf.train.AdamOptimizer(learning_rate=self.tf_lr, beta1=0.5, beta2=0.9).minimize(loss, var_list=params)
                            optimizers.append(optimizer)
                            cd[gpu].append(optimizer)
                        gpu  = (gpu + 1) % self.nr_gpu            

            total_optimizer = tf.group(self.maintain_averages_op, *optimizers)
            total_loss      = tf.add_n(losses)
                
        if not init:
            return total_loss, total_optimizer
        
    def initialize(self, x_init, y_init=None):
        if self.class_conditional:
            feed_dict={self.x_init: x_init, self.y_init: y_init, self.tf_lr:0.001,
                       self.x: x_init, self.y: y_init}
        else:
            feed_dict={self.x_init: x_init, self.tf_lr:0.001,
                       self.x: x_init}

        self.sess.run(self.initializer, feed_dict =feed_dict)
        self.sess.run(self.adam_init, feed_dict = feed_dict)
        print('Model initialized')
        
    def inference(self):
        cd = [None] * self.nr_gpu
        gpu = 0

        x_in       = [None] * (self.bit_len * self.channel)
        cond_input = [None] * (self.bit_len * self.channel)
        x_out      = [None] * (self.bit_len * self.channel)
        
        if self.class_conditional:
            y_in       = tf.placeholder(tf.int32, shape=(self.batch_size,))
            h_in       = tf.one_hot(y_in, depth=self.num_labels)
        else:
            y_in = None
            h_in = None
        
        for i in range(self.bit_len):
            for c in range(self.channel):
                with tf.device('/gpu:%d' % gpu):
                    with tf.control_dependencies(cd[gpu]):
                        name = 'model_bit_' + str(i) + '_color' + str(c)
                        
                        x_in[self.channel*i + c] = tf.placeholder(tf.float32, shape=(self.batch_size,)  + (self.dim, self.dim, 1))
                        if self.channel*i + c > 0:
                            cond_input[self.channel*i + c] = tf.placeholder(tf.float32, shape=(self.batch_size,) + (self.dim, self.dim, self.channel*i + c))
                            x_input = tf.concat([cond_input[self.channel*i + c], x_in[self.channel*i + c]], 3)
                        else:
                            x_input = x_in[self.channel*i + c]                                                

                        out = model_spec(x_input, h=h_in, cond=cond_input[self.channel*i + c], reuse=True, ema=None, 
                                             dropout_p=self.dropout_p, scope=name, **self.model_opt)

                        out_sample = nn.sample_from_bernoulli(out)
                        x_out[self.channel*i + c] = out_sample
                    gpu = (gpu+1) % self.nr_gpu

        self.x_in = x_in
        self.cond_input = cond_input
        self.x_out = x_out
        self.y_in = y_in
        
    def train(self, x, y=None, lr=0.01):
        if self.class_conditional:
            feed_dict = {self.x:x, self.y:y, self.tf_lr:lr}
        else:
            feed_dict = {self.x:x, self.tf_lr:lr}
        loss = self.sess.run([self.total_loss, self.total_optimizer], feed_dict)[0]
        return loss
        
    def check_and_restore_model(self, ckpt_file):
        print('restoring params from', ckpt_file)
        self.saver.restore(self.sess, ckpt_file)    
        
    def save_model(self, save_dir):
        self.saver.save(self.sess, save_dir)

    def sample(self, y=None):
        x_cond = None
        
        for i in range(self.bit_len):
            for c in range(self.channel):
                if self.channel*i + c > 0:
                    if self.class_conditional:
                        feed_dict = {self.cond_input[self.channel*i + c]: x_cond, self.y_in: y}
                    else:
                        feed_dict = {self.cond_input[self.channel*i + c]: x_cond}
                else:
                    if self.class_conditional:
                        feed_dict = {self.y_in: y}
                    else:
                        feed_dict = {}
                        
                x_current = np.zeros([self.batch_size, self.dim, self.dim, 1])
                for y_i in range(self.dim):
                    for x_i in range(self.dim):
                        feed_dict.update({self.x_in[self.channel*i + c]: x_current})
                        x_current_new = self.sess.run(self.x_out[self.channel*i + c], feed_dict=feed_dict)
                        x_current[:, y_i, x_i, :] = x_current_new[:, y_i, x_i, :]
                if self.channel*i + c > 0:
                    x_cond = np.concatenate([x_cond, x_current], 3)
                else:
                    x_cond = x_current
    
        x_out = np.reshape(x_cond, [self.batch_size, self.dim, self.dim, self.bit_len, self.channel])
        x_out = np.transpose(x_out, [0, 1, 2, 4, 3])

        
        """ add bit_map to numeric """
        x_out = U.bit_to_num(x_out, self.bit_len)
        return x_out
        
        
    