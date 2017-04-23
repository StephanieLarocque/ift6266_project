
# coding: utf-8

# In[1]:

#Model classes inspired from Francis Dutil
#Other references
#https://openai.com/blog/generative-models/
#https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
import lasagne
import theano
import theano.tensor as T
import numpy as np

#Import layers from lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer
from lasagne.layers import BatchNormLayer , batch_norm
from lasagne.layers import NonlinearityLayer, DimshuffleLayer, ConcatLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import Layer, PadLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Deconv2DLayer #same as TransposeConv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GaussianNoiseLayer
from lasagne.layers import FlattenLayer

#Import nonlinearities
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import tanh
from lasagne.nonlinearities import LeakyRectify


# In[2]:

class Model(object):
    def __init__(self):
        pass

    def build_network(self):
        raise NotImplementedError





# In[5]:

class discriminator(Model):
    def __init__(self):
        pass

    def build_network(self,
                input_var,
                contour_var,
                conv_before_pool = [2],
                n_filters = 32,
                filter_size = 3,
                n_units_dense_layer = 256,
                out_nonlin = sigmoid):
        
        
        self.conv_before_pool = conv_before_pool
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_units_dense_layer = n_units_dense_layer
        self.out_nonlin = out_nonlin

        
        self.contour_var = contour_var
        #assert len(conv_before_pool)==4
        n_block = len(conv_before_pool)
        net = {}


        # net['fake_input'] = InputLayer((None, 3, 32, 32), fake_input_var)
        # net['true_input'] = InputLayer((None, 3, 32, 32), true_input_var)
        # net['full_input'] = ConcatLayer([net['fake_input'], net['true_input']], axis=0)
        # incoming_layer = 'full_input'

        net['input'] = InputLayer((None,3,32,32), input_var)
        incoming_layer = 'input'
        
        net['noise_to_input'] = GaussianNoiseLayer(net[incoming_layer], sigma=1.0)
        incoming_layer = 'noise_to_input'

        
        net['pad_input'] = PadLayer(net[incoming_layer], width=16, val=0)
        incoming_layer = 'pad_input'
        
        net['contour'] = InputLayer((None, 3, 64, 64), contour_var)
        incoming_layer ='contour'
        
        net['in+out'] = ElemwiseSumLayer([net['pad_input'], net['contour']])
        incoming_layer = 'in+out'
        


        for i in range(n_block):
            n_conv = conv_before_pool[i]

            for c in range(n_conv):
                #Do we use deconv2d or simply conv2d ?
                #Is there any difference?
                #Its a transpose convolutionlayer...
                net['conv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
                            num_filters = n_filters*(2**i),
                            filter_size = filter_size,
                            pad = 2,
                            stride = 2,
                            nonlinearity = None)
                incoming_layer = 'conv'+str(i)+'_'+str(c)

                net['bn'+str(i)+'_'+str(c)] = BatchNormLayer(net[incoming_layer])
                incoming_layer = 'bn'+str(i)+'_'+str(c)


                net['nonlin'+str(i)+'_'+str(c)]=NonlinearityLayer(
                            net[incoming_layer],
                            nonlinearity = LeakyRectify(0.2))
                incoming_layer = 'nonlin'+str(i)+'_'+str(c)

#            if i<n_block-1:
#                #Pooling layer
#                net['pool'+str(i)] = Pool2DLayer(net[incoming_layer],
#                                pool_size = 2, mode = 'average_exc_pad')
#                incoming_layer = 'pool'+str(i)

#        #Add 1 dense layer
#        net['dense'] = DenseLayer(net[incoming_layer],
#                            num_units = n_units_dense_layer)
#        incoming_layer = 'dense'


        net['flatten']= FlattenLayer(net[incoming_layer])
        incoming_layer = 'flatten'

        #Last layer must have 1 units (binary classification)
        net['last_layer'] = DenseLayer(net[incoming_layer],
                            num_units = 1,
                            nonlinearity = None)
        incoming_layer = 'last_layer'
        
        net['real_last_layer'] = NonlinearityLayer(net[incoming_layer], nonlinearity = sigmoid)
        incoming_layer = 'real_last_layer'


        self.net = net[incoming_layer]
        self.dict_net = net


# In[ ]:




# In[6]:

class discriminator_over_generator(Model):
    def __init__(self):
        pass

    def build_network(self,
                G_net,
                D_net):

        
        conv_before_pool = D_net.conv_before_pool
        n_filters = D_net.n_filters
        filter_size = D_net.filter_size
        n_units_dense_layer = D_net.n_units_dense_layer
        out_nonlin = D_net.out_nonlin
        
        contour_var = D_net.contour_var
        
  
        n_block = len(conv_before_pool)
        net = {}


        #TODO : faire un layer qui ne fait rien!!!
        net['input'] = Pool2DLayer(G_net.net, pool_size=1)
        incoming_layer = 'input'
        


        
        net['pad_input'] = PadLayer(net[incoming_layer], width=16, val=0)
        incoming_layer = 'pad_input'
        
        net['contour'] = InputLayer((None, 3, 64, 64), contour_var)
        incoming_layer ='contour'
        
        net['in+out'] = ElemwiseSumLayer([net['pad_input'], net['contour']])
        incoming_layer = 'in+out'
        
        



        for i in range(n_block):
            n_conv = conv_before_pool[i]

            for c in range(n_conv):
                #Do we use deconv2d or simply conv2d ?
                #Is there any difference?
                #Its a transpose convolutionlayer...
                net['conv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
                            num_filters = n_filters*(2**i),
                            filter_size = filter_size,
                            pad = 2,
                            stride = 2,                               
                            W = D_net.dict_net['conv'+str(i)+'_'+str(c)].W,
                            b = D_net.dict_net['conv'+str(i)+'_'+str(c)].b,
                            nonlinearity = None)
                incoming_layer = 'conv'+str(i)+'_'+str(c)


                net['bn'+str(i)+'_'+str(c)] = BatchNormLayer(net[incoming_layer],
                            beta=D_net.dict_net['bn'+str(i)+'_'+str(c)].beta,
                            gamma=D_net.dict_net['bn'+str(i)+'_'+str(c)].gamma)
                incoming_layer = 'bn'+str(i)+'_'+str(c)

                net['nonlin'+str(i)+'_'+str(c)]=NonlinearityLayer(
                            net[incoming_layer],
                            nonlinearity = LeakyRectify(0.2))
                incoming_layer = 'nonlin'+str(i)+'_'+str(c)


        
        net['flatten']= FlattenLayer(net[incoming_layer])
        incoming_layer = 'flatten'
        #Last layer must have 1 units (binary classification)
        net['last_layer'] = DenseLayer(net[incoming_layer],
                            num_units = 1,
                            nonlinearity = None,
                            W = D_net.dict_net['last_layer'].W,
                            b = D_net.dict_net['last_layer'].b)
        incoming_layer = 'last_layer'
        
        net['real_last_layer'] = NonlinearityLayer(net[incoming_layer], nonlinearity = sigmoid)
        incoming_layer = 'real_last_layer'

        #Softmax layer needed somehere
        self.net = net[incoming_layer]
        self.dict_net = net



if __name__=='__main__':

    G_input_var = T.tensor4('G_input')
    D_input_var = T.tensor4('D_input_true')


    GAN = gan(G_input_var, D_input_var)

    print 'generator'
    print lasagne.layers.get_all_layers(GAN.G)
    print 'discriminator'
    print lasagne.layers.get_all_layers(GAN.D)
    print 'discriminator_over_generator'
    print lasagne.layers.get_all_layers(GAN.D_over_G)

    #TODO : verifier que les valeurs sont les memes
    #TODO : doit on reassigne les parametres a D_over_G ou updater
    #D updatera aussi D_over_G ???
