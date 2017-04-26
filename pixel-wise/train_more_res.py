"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import tqdm
import argparse

import numpy as np
import tensorflow as tf

from pixel_cnn_pp.model_bitwise import PxCNN_bw
import pixel_cnn_pp.utils as U
import pixel_cnn_pp.plotting as plotting
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp-bitwise/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp-bitwise/bitwise/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=10, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('--load_epoch', type=int, default=0, help='What epoch number to restore?')
parser.add_argument('--sample_means', action='store_true', help='Biased sampling?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=4, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,  help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size',      type=int, default=40, help='Batch size during training per GPU')
parser.add_argument('-a', '--batch_size_init', type=int, default=40, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
DataLoader = {'cifar': cifar10_data.DataLoader, 'imagenet': imagenet_data.DataLoader}[args.data_set]

train_data = DataLoader(args.data_dir, 'train', args.batch_size, rng=rng, shuffle=True, return_labels=True)
test_data = DataLoader(args.data_dir, 'test',   args.batch_size, shuffle=False, return_labels=True)


num_labels = train_data.get_num_labels()
bit_len = 3
dim = 32

model = PxCNN_bw(nr_resnet = args.nr_resnet, 
                 nr_filters = args.nr_filters, 
                 resnet_nonlinearity = 'concat_tanh', 
#                 class_conditional   = True, 
                 class_conditional = False,
                 dim = dim, 
                 channel = 3, 
                 bit_len = bit_len,
                 batch_size = args.batch_size, 
                 batch_size_init = args.batch_size_init, 
                 dropout_p = args.dropout_p, 
                 nr_gpu = args.nr_gpu, 
                 num_labels = num_labels,
                 polyak_decay = args.polyak_decay)


# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')


lr = args.learning_rate
losses = []


for epoch in range(args.max_epochs):
    begin = time.time()
    
    if epoch == 0:
        inputs_init, labels_init = train_data.next(args.batch_size_init)
        inputs_init, labels_init = U.make_feed_data((inputs_init, labels_init), bit_len)
        train_data.reset()
        model.initialize(x_init=inputs_init )
        
        if args.load_params:
            ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
            model.check_and_restore_model(ckpt_file)
            
            
    # train for one epoch
    train_losses = []
    
    
    for d in train_data:
        inputs, labels = U.make_feed_data(d, bit_len)
        lr *= args.lr_decay
        loss = model.train(x=inputs, lr=lr)
        train_losses.append(loss)
        
    train_loss = np.mean(train_losses) / (np.log(2) * args.batch_size * dim * dim * 3)
    print('Epoch %d, time=%ds, train logloss = %.4f.' % (epoch, time.time() - begin, train_loss))
    sys.stdout.flush()       
    
    losses.append(train_loss)
    np.savez(args.save_dir + '/train_losses_' + args.data_set + '.npz', train_loss = losses)
    
    
    save_dir = args.save_dir + '/params_' + args.data_set + '.ckpt'

    """ sampling """
    if epoch % args.save_interval == 0 and epoch > 0:
        count=0
        for d in test_data:
            if count == 4:
                break
            count +=1 
            inputs, labels = d
            gen_begin = time.time()
            outputs = model.sample() #y=labels)
            
            img_tile = plotting.img_tile(outputs[:int(np.floor(np.sqrt(args.batch_size))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir, '%s_sample%d.png' % (args.data_set, epoch)))
            plotting.plt.close('all')
    
    if epoch % 100 == 0 and epoch>0:
        model.save_model(save_dir)
    