#Model classes inspired from Francis Dutil
#Other references
#https://openai.com/blog/generative-models/
#https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
import lasagne
import theano
import theano.tensor as T
import numpy as np

#Import layers from lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer, \
        NonlinearityLayer, DimshuffleLayer, ConcatLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Deconv2DLayer #same as TransposeConv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import DenseLayer

#Import nonlinearities
from lasagne.nonlinearities import rectify


class Model(object):
    def __init__(self):
        pass

    def build_network(self):
        raise NotImplementedError


class generator(Model):
    def __init__(self):
        pass

    def build_network(self, input_var,
                    #Maybe input_var not necessary because it's random noise
                    conv_before_pool=[1,1,1,1],
                    n_filters = 64, #number of filters before final deconv layer
                    filter_size = 3
                    ):

        # self.conv_before_pool = conv_before_pool
        # self.n_filters = n_filters
        # self.filter_size = filter_size
        assert len(conv_before_pool)==4 #to recover the right image size
        assert min(conv_before_pool)>=1 #must have at least 1 convolution in each block
        n_block = len(conv_before_pool)

        net = {}
        #input_var = ? will be (for example) a 100x1 latent variables
        #For now : starts with 4x4 images and do 3 upscaling layers
        #so it finishes with 32x32 images
        net['input'] = InputLayer((None, n_filters*2**(n_block-1), 4, 4), input_var)
        incoming_layer = 'input'

        #TODO : comment generer le bruit initial
        for i in range(n_block):
            n_conv = conv_before_pool[i]

            for c in range(n_conv):
                #c-th convolution for the i-th block
                net['conv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
                        #n_filters decreases as size of image increases
                        num_filters = n_filters * 2**(n_block-i-1),
                        filter_size = filter_size,
                        nonlinearity = rectify,
                        pad='same')
                incoming_layer = 'conv'+str(i)+'_'+str(c)

            #Upscaling layer (except when the image is already 32x32)
            if i<n_block-1:
                net['upscale'+str(i)] = Upscale2DLayer(net[incoming_layer],
                            scale_factor = 2, mode='repeat')
                incoming_layer = 'upscale'+str(i)


        #Final convolution to recover the size 32x32x3
        net['finalconv'] = Conv2DLayer(net[incoming_layer],
                        num_filters = 3, #RGB filters
                        filter_size = filter_size,
                        nonlinearity = rectify,
                        pad = 'same')
        incoming_layer = 'finalconv'

        return net, incoming_layer


class discriminator(Model):
    def __init__(self):
        pass

    def build_network(self, input_var,
                conv_before_pool = [1,1,1,1],
                n_filters = 64,
                filter_size = 3,
                n_units_dense_layer = 1024):

        assert len(conv_before_pool)==4
        n_block = len(conv_before_pool)
        net = {}

        net['input'] = InputLayer((None, 3, 32, 32), input_var)
        incoming_layer = 'input'


        for i in range(n_block):
            n_conv = conv_before_pool[i]

            for c in range(n_conv):
                #Do we use deconv2d or simply conv2d ?
                #Is there any difference?
                #Its a transpose convolutionlayer...
                net['conv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
                            num_filters = n_filters*(2**i),
                            filter_size = filter_size,
                            pad = 'same')
                incoming_layer = 'conv'+str(i)+'_'+str(c)

            if i<n_block-1:
                #Pooling layer
                net['pool'+str(i)] = Pool2DLayer(net[incoming_layer],
                                pool_size = 2)
                incoming_layer = 'pool'+str(i)

        #Add 1 dense layer
        net['dense'] = DenseLayer(net[incoming_layer],
                            num_units = n_units_dense_layer)
        incoming_layer = 'dense'

        #Last layer must have 1 units (binary classification)
        net['last_layer'] = DenseLayer(net[incoming_layer], num_units = 1)
        incoming_layer = 'last_layer'

        #Softmax layer needed somehere

        return net, incoming_layer





# class cond_gan(Model):
#     def __init__(self):
#         pass
#
#     def build_network(self):
#         pass


if __name__=='__main__':
    pass
