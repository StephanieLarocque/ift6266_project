
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
from lasagne.layers import Layer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Deconv2DLayer #same as TransposeConv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import DenseLayer

#Import nonlinearities
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import sigmoid


# In[2]:

class Model(object):
    def __init__(self):
        pass

    def build_network(self):
        raise NotImplementedError


# In[3]:

class gan():
    def __init__(self, G_input_var,
                    D_input_var,
                    conv_before_pool=[1,1,1,1],
                    n_filters = 64, #number of filters before final deconv layer
                    filter_size = 3,
                    n_units_dense_layer = 1024,
                    out_nonlin = sigmoid):

        self.G  = generator().build_network(
                    G_input_var,
                    conv_before_pool = conv_before_pool,
                    n_filters = n_filters,
                    filter_size=filter_size)
        self.D = discriminator().build_network(
                    D_input_var,
                    conv_before_pool = conv_before_pool,
                    n_filters = n_filters,
                    filter_size=filter_size,
                    n_units_dense_layer =n_units_dense_layer,
                    out_nonlin =out_nonlin)
        self.D_over_G = discriminator_over_generator().build_network(
                    self.G,
                    self.D,
                    conv_before_pool = conv_before_pool,
                    n_filters = n_filters,
                    filter_size=filter_size,
                    n_units_dense_layer=n_units_dense_layer,
                    out_nonlin = out_nonlin)




# In[ ]:




# In[4]:

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
        net['last_layer'] = Conv2DLayer(net[incoming_layer],
                        num_filters = 3, #RGB filters
                        filter_size = filter_size,
                        nonlinearity = rectify,
                        pad = 'same')
        incoming_layer = 'last_layer'
        #self.last_layer = incoming_layer

        return net


# In[ ]:




# In[5]:

class discriminator(Model):
    def __init__(self):
        pass

    def build_network(self,
                input_var,
                conv_before_pool = [1,1,1,1],
                n_filters = 64,
                filter_size = 3,
                n_units_dense_layer = 1024,
                out_nonlin = sigmoid):

        assert len(conv_before_pool)==4
        n_block = len(conv_before_pool)
        net = {}


        # net['fake_input'] = InputLayer((None, 3, 32, 32), fake_input_var)
        # net['true_input'] = InputLayer((None, 3, 32, 32), true_input_var)
        # net['full_input'] = ConcatLayer([net['fake_input'], net['true_input']], axis=0)
        # incoming_layer = 'full_input'

        net['input'] = InputLayer((None,3,32,32), input_var)
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
                            pad = 'same',
                            nonlinearity = None)
                incoming_layer = 'conv'+str(i)+'_'+str(c)

                net['bn'+str(i)+'_'+str(c)] = BatchNormLayer(net[incoming_layer])
                incoming_layer = 'bn'+str(i)+'_'+str(c)


                net['nonlin'+str(i)+'_'+str(c)]=NonlinearityLayer(
                            net[incoming_layer],
                            nonlinearity = rectify)
                incoming_layer = 'nonlin'+str(i)+'_'+str(c)

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
        net['last_layer'] = DenseLayer(net[incoming_layer],
                            num_units = 1,
                            nonlinearity = out_nonlin)
        incoming_layer = 'last_layer'

        #Softmax layer needed somehere

        return net


# In[ ]:




# In[6]:

class discriminator_over_generator(Model):
    def __init__(self):
        pass

    def build_network(self,
                G_net,
                D_net,
                conv_before_pool = [1,1,1,1],
                n_filters = 64,
                filter_size = 3,
                n_units_dense_layer = 1024,
                out_nonlin = sigmoid):

        assert len(conv_before_pool)==4
        n_block = len(conv_before_pool)
        net = {}


        #TODO : faire un layer qui ne fait rien!!!
        net['input'] = Pool2DLayer(G_net['last_layer'], pool_size=1)
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
                            pad = 'same',
                            W = D_net['conv'+str(i)+'_'+str(c)].W,
                            b = D_net['conv'+str(i)+'_'+str(c)].b,
                            nonlinearity = None)
                incoming_layer = 'conv'+str(i)+'_'+str(c)


                net['bn'+str(i)+'_'+str(c)] = BatchNormLayer(net[incoming_layer],
                            beta=D_net['bn'+str(i)+'_'+str(c)].beta,
                            gamma=D_net['bn'+str(i)+'_'+str(c)].gamma)
                incoming_layer = 'bn'+str(i)+'_'+str(c)

                net['nonlin'+str(i)+'_'+str(c)]=NonlinearityLayer(
                            net['bn'+str(i)+'_'+str(c)],
                            nonlinearity = rectify)
                incoming_layer = 'nonlin'+str(i)+'_'+str(c)



            if i<n_block-1:
                #Pooling layer
                net['pool'+str(i)] = Pool2DLayer(net[incoming_layer],
                                pool_size = 2)
                incoming_layer = 'pool'+str(i)

        #Add 1 dense layer
        net['dense'] = DenseLayer(net[incoming_layer],
                            num_units = n_units_dense_layer,
                            W = D_net['dense'].W,
                            b = D_net['dense'].b)
        incoming_layer = 'dense'

        #Last layer must have 1 units (binary classification)
        net['last_layer'] = DenseLayer(net[incoming_layer],
                            num_units = 1,
                            nonlinearity = out_nonlin,
                            W = D_net['last_layer'].W,
                            b = D_net['last_layer'].b)
        incoming_layer = 'last_layer'

        #Softmax layer needed somehere

        return net



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
