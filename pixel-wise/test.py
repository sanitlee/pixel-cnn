from pixel_cnn_pp.model_bitwise import PxCNN_bw
import pixel_cnn_pp.plotting as plotting
import data.cifar10_data as cifar10_data
import numpy as np
import time
import pixel_cnn_pp.utils as U
import os


def test_bit_transform():
    DataLoader = cifar10_data.DataLoader
    train_data = DataLoader('/tmp/pxpp-bitwise/data', subset='train', batch_size=20)
    x = train_data.next(20)
    x_scaled = U.scale_function(x)


    d_ = 32
    x_3bit = (np.floor(x.astype(np.float32) / d_) * d_) / ((255 // d_) * d_)
    x_3bit = 2. * x_3bit - 1.
    
    
    bits = U.num_to_bit(x, 3)
    recov = U.bit_to_num(bits, 3)

    img_tile = plotting.img_tile(x_scaled, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile)
    save_dir = '/tmp/paralel-px/save'
    plotting.plt.savefig(os.path.join(save_dir,'test1.png' ))
    plotting.plt.close('all')
    
    img_tile = plotting.img_tile(x_3bit, aspect_ratio=1.0, border_color=1.0, stretch =True)
    img = plotting.plot_img(img_tile)
    save_dir = '/tmp/paralel-px/save'
    plotting.plt.savefig(os.path.join(save_dir,'test2.png' ))
    plotting.plt.close('all')
    
    recov_scaled = recov /255.
    img_tile = plotting.img_tile(recov_scaled, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile)
    save_dir = '/tmp/paralel-px/save'
    plotting.plt.savefig(os.path.join(save_dir,'test3.png' ))
    plotting.plt.close('all')    


    


def test_model():
    dim_max =  16
    batch_size_init = 5
    batch_size = 5
    nr_filters = 64
    nr_resnet = 1
    dropout_p = 0.5
    channel = 3
    nr_gpu = 8
    polyak_decay = 0.9995
    max_epoch = 20
    bit_len = 3

    # test initializer
    print('test initialization')
    
    imgs_init = np.random.randint(0, 2**bit_len, size=(batch_size_init, dim_max, dim_max, channel))
    imgs      = np.random.randint(0, 2**bit_len, size=(batch_size ,     dim_max, dim_max, channel))
    
    labels_init = np.random.randint(0, 10, size=(batch_size_init))
    labels      = np.random.randint(0, 10, size=(batch_size))
    
    
    imgs_init, labels_init = U.make_feed_data((imgs_init, labels_init), bit_len)
    imgs,      labels      = U.make_feed_data((imgs, labels), bit_len)
    
    model = PxCNN_bw(nr_resnet = nr_resnet, 
                     nr_filters = nr_filters, 
                     resnet_nonlinearity = 'concat_tanh', 
                     class_conditional   = False,  
                     dim = dim_max, 
                     channel = 3, 
                     bit_len = bit_len,
                     batch_size = batch_size, 
                     batch_size_init = batch_size_init, 
                     dropout_p = dropout_p, 
                     nr_gpu = nr_gpu, 
                     num_labels = 10,
                     polyak_decay = polyak_decay)
    
    model.initialize(x_init = imgs_init)# , y_init = labels_init)
    
    lr = 0.001
    print('test training')
    for i in range(max_epoch):
        lr *= polyak_decay
        accu = model.train(x = imgs,lr=lr) # y=labels)
        
        print('Iteration %d: loss is %.4f.' % (i, accu))
        
    print('test sampling')
    begin = time.time()
    sample = model.sample() #y=labels)
    print('sample time is %.4f'%(time.time() - begin))
    print('sample shape')
    print(sample.shape)
    
    print('test plotting')
    save_dir = '/tmp/pxpp-bitwise/bitwise/save'
    img_tile=plotting.img_tile(sample[:int(np.floor(np.sqrt(batch_size))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='test samples')
    plotting.plt.savefig(os.path.join(save_dir, 'test_sample%d.png'))
    plotting.plt.close('all')

    return 

if __name__ == '__main__':
#     test_model()
     test_bit_transform()
    