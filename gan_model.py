
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
                input_var, #center de l'image/inpainting
                contour_var, #contour de l'image
                all_image = True,
                conv_before_pool = [2],
                n_filters = 32,
                filter_size = 7,
                #n_units_dense_layer = 256,
                out_nonlin = sigmoid):
        
        
        self.conv_before_pool = conv_before_pool
        self.n_filters = n_filters
        self.filter_size = filter_size
        #self.n_units_dense_layer = n_units_dense_layer
        self.out_nonlin = out_nonlin
        self.all_image = all_image

        
        self.contour_var = contour_var
        #assert len(conv_before_pool)==4
        n_block = len(conv_before_pool)
        net = {}


        net['input'] = InputLayer((None,3,32,32), input_var)
        incoming_layer = 'input'
        
        #net['noise_to_input'] = GaussianNoiseLayer(net[incoming_layer], sigma=0.1)
        #incoming_layer = 'noise_to_input'

        if all_image:
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
                            pad = 'same',
                            #stride = 2,
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
            net['pool'+str(i)] = Pool2DLayer(net[incoming_layer], pool_size=2)
            incoming_layer = 'pool'+str(i)




        net['flatten']= FlattenLayer(net[incoming_layer])
        incoming_layer = 'flatten'

        #Last layer must have 1 units (binary classification)
        net['last_layer'] = DenseLayer(net[incoming_layer],
                            num_units = 1,
                            nonlinearity = None)
        incoming_layer = 'last_layer'
        
        #net['final_nonlin'] = NonlinearityLayer(net[incoming_layer], nonlinearity = sigmoid)
        #incoming_layer = 'final_nonlin'
        


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
        #n_units_dense_layer = D_net.n_units_dense_layer
        out_nonlin = D_net.out_nonlin
        all_image = D_net.all_image
        
        contour_var = D_net.contour_var
        
  
        n_block = len(conv_before_pool)
        net = {}


        #TODO : faire un layer qui ne fait rien!!!
        net['input'] = Pool2DLayer(G_net.net, pool_size=1)
        incoming_layer = 'input'
        
        
        if all_image:
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
                            pad = 'same',
                            #stride = 2,                               
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
                
            net['pool'+str(i)] = Pool2DLayer(net[incoming_layer], pool_size=2)
            incoming_layer = 'pool'+str(i)


        
        net['flatten']= FlattenLayer(net[incoming_layer])
        incoming_layer = 'flatten'
        #Last layer must have 1 units (binary classification)
        net['last_layer'] = DenseLayer(net[incoming_layer],
                            num_units = 1,
                            W = D_net.dict_net['last_layer'].W,
                            b = D_net.dict_net['last_layer'].b,
                            nonlinearity=None)
        incoming_layer = 'last_layer'
        
        #net['final_nonlin'] = NonlinearityLayer(net[incoming_layer], nonlinearity = sigmoid)
        #incoming_layer = 'final_nonlin'
        

        #Softmax layer needed somehere
        self.net = net[incoming_layer]
        self.dict_net = net

