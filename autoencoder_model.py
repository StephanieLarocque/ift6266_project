import lasagne
import theano
import theano.tensor as T
import numpy as np

#Import layers from lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer
from lasagne.layers import BatchNormLayer, batch_norm
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


class Model(object):
    def __init__(self):
        pass

    def build_network(self):
        raise NotImplementedError


class AE_contour2center(Model):
    def __init__(self):
	pass
	
    def build_network(self, input_var,
                      conv_before_pool=[2,2],
                      n_filters = 32,
                      code_size = 100,
                      filter_size = 3,
                      pool_factor = 2):
		
	self.conv_before_pool = conv_before_pool
	self.code_size = code_size
	self.n_filters = n_filters
	n_block = len(conv_before_pool)
	
	net={}
	#Input var = whole image with center removed
	net['input'] = InputLayer((None, 3, 64, 64), input_var)
	incoming_layer = 'input'
		
	#Encoder (Conv + Pooling Layers)
	for i in range(n_block): 
	    n_conv = conv_before_pool[i]
		
	    for c in range(n_conv): #c-th convolution for the i-th block
		net['conv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
		            #n_filters increases as size of image decreases
			    num_filters = n_filters * pool_factor**i,
			    filter_size = filter_size,
			    nonlinearity = rectify,
			    pad='same')
                incoming_layer = 'conv'+str(i)+'_'+str(c)
                net['bn'+str(i)+'_'+str(c)] = batch_norm(net[incoming_layer])
                incoming_layer = 'bn'+str(i)+'_'+str(c)
                
                
	    net['pool'+str(i)] = Pool2DLayer(net[incoming_layer], pool_size = pool_factor)
	    incoming_layer = 'pool'+str(i)
            
        #Encoder (Reshape + Dense Layer)
        net['reshape_enc'] = ReshapeLayer(net[incoming_layer],
        	shape=([0],-1))
        incoming_layer = 'reshape_enc'
        n_units_before_dense = net[incoming_layer].output_shape[1]
        
        #Code layer
        net['code_dense'] = DenseLayer(net[incoming_layer], num_units = code_size)
        incoming_layer = 'code_dense'

        #Decoder (Dense + Reshape Layer)
        net['dense_up'] = DenseLayer(net[incoming_layer], num_units = n_units_before_dense)
        incoming_layer = 'dense_up'
        
        net['reshape_dec'] = ReshapeLayer(net[incoming_layer],
                shape=([0],-1,n_filters/(2**(n_block-1)), n_filters/(2**(n_block-1)) ))
        incoming_layer = 'reshape_dec'


        #Decoder (Upscaling + Conv Layers)
        #Must recover the inpainting (only 32x32)

        for i in range(n_block-1,0,-1):
            net['upscale'+str(i)] = Upscale2DLayer(net[incoming_layer], scale_factor = pool_factor)
	    incoming_layer = 'upscale'+str(i)
            
	    for c in range(n_conv-1,-1,-1): #c-th convolution for the i-th block
                net['upconv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
		            #n_filters increases as size of image decreases
			    num_filters = n_filters * pool_factor**i,
			    filter_size = filter_size,
			    nonlinearity = rectify,
			    pad='same')
                incoming_layer = 'upconv'+str(i)+'_'+str(c)
                net['bn'+str(i)+'_'+str(c)] = batch_norm(net[incoming_layer])
                incoming_layer = 'bn'+str(i)+'_'+str(c)
            
        net['last_layer'] = Conv2DLayer(net[incoming_layer],
                num_filters = 3,
                filter_size = 1,
                pad='same')
        incoming_layer = 'last_layer'
                
        
       
        return net, incoming_layer




class AE_center2center(Model):
    def __init__(self):
	pass
	
    def build_network(self, input_var,
                      conv_before_pool=[2,2],
                      n_filters = 32,
                      code_size = 100,
                      filter_size = 3,
                      pool_factor = 2):
		
	self.conv_before_pool = conv_before_pool
	self.code_size = code_size
	self.n_filters = n_filters
	n_block = len(conv_before_pool)
	
	net={}
	net['input'] = InputLayer((None, 3, 32, 32), input_var)
	incoming_layer = 'input'
		
	#Encoder (Conv + Pooling Layers)
	for i in range(n_block):
	    n_conv = conv_before_pool[i]
		
	    for c in range(n_conv): #c-th convolution for the i-th block
		net['conv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
		            #n_filters increases as size of image decreases
			    num_filters = n_filters * pool_factor**i,
			    filter_size = filter_size,
			    nonlinearity = rectify,
			    pad='same')
                incoming_layer = 'conv'+str(i)+'_'+str(c)
                net['bn'+str(i)+'_'+str(c)] = batch_norm(net[incoming_layer])
                incoming_layer = 'bn'+str(i)+'_'+str(c)
                
                
	    net['pool'+str(i)] = Pool2DLayer(net[incoming_layer], pool_size = pool_factor)
	    incoming_layer = 'pool'+str(i)
            
        #Encoder (Reshape + Dense Layer)
        net['reshape_enc'] = ReshapeLayer(net[incoming_layer],
        	shape=([0],-1))
        incoming_layer = 'reshape_enc'
        n_units_before_dense = net[incoming_layer].output_shape[1]
        
        #Code layer
        net['code_dense'] = DenseLayer(net[incoming_layer], num_units = code_size)
        incoming_layer = 'code_dense'

        #Decoder (Dense + Reshape Layer)
        net['dense_up'] = DenseLayer(net[incoming_layer], num_units = n_units_before_dense)
        incoming_layer = 'dense_up'
        
        net['reshape_dec'] = ReshapeLayer(net[incoming_layer],
                shape=([0],-1,n_filters/(2**n_block), n_filters/(2**n_block) ))
        incoming_layer = 'reshape_dec'


        #Decoder (Upscaling + Conv Layers)

        for i in range(n_block-1,-1,-1):
            net['upscale'+str(i)] = Upscale2DLayer(net[incoming_layer], scale_factor = pool_factor)
	    incoming_layer = 'upscale'+str(i)
            
	    for c in range(n_conv-1,-1,-1): #c-th convolution for the i-th block
                net['upconv'+str(i)+'_'+str(c)] = Conv2DLayer(net[incoming_layer],
		            #n_filters increases as size of image decreases
			    num_filters = n_filters * pool_factor**i,
			    filter_size = filter_size,
			    nonlinearity = rectify,
			    pad='same')
                incoming_layer = 'upconv'+str(i)+'_'+str(c)
                net['bn'+str(i)+'_'+str(c)] = batch_norm(net[incoming_layer])
                incoming_layer = 'bn'+str(i)+'_'+str(c)
            
        net['last_layer'] = Conv2DLayer(net[incoming_layer],
                num_filters = 3,
                filter_size = 1,
                pad='same')
        incoming_layer = 'last_layer'
                
        
	 

        
       
        return net, incoming_layer
            
if __name__ == '__main__':
    input_var = T.tensor4('input')
    ae=AE().build_network(input_var)
    layers = lasagne.layers.get_all_layers(ae)
    for l in layers :
        print l, ae[l], ae[l].output_shape
		
	
	
	
	
	
	
	
