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

from matplotlib import pyplot as plt



class Model(object):
    def __init__(self):
        pass

    def build_network(self):
        raise NotImplementedError


    def compile_theano_functions(self, learning_rate = 0.001):


        input_var = self.input_var#T.tensor4('input img bx3x64x64')
        target_var = T.tensor4('inpainting target')

        print "Defining and compiling train functions"

        pred_img = lasagne.layers.get_output(self.net)
        loss = self.get_loss(pred_img, target_var)
        params = lasagne.layers.get_all_params(self.net, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate = learning_rate)
        train_fn = theano.function([input_var,target_var], loss, updates = updates,
                                    allow_input_downcast=True)
        self.train_fn = train_fn


        print "Defining and compiling valid functions"

        valid_pred_imgs = lasagne.layers.get_output(self.net,deterministic=True)
        valid_loss = self.get_loss(valid_pred_imgs, target_var)
        valid_fn = theano.function([input_var, target_var], valid_loss, allow_input_downcast=True)
        self.valid_fn = valid_fn

        print 'Defining and compiling get_imgs function'

        self.get_imgs = theano.function([input_var], lasagne.layers.get_output(self.net,deterministic = True),
                          allow_input_downcast=True)
        print "Done"



    def get_loss(self, prediction, target):
        return T.mean(lasagne.objectives.squared_error(prediction, target))


    def extract_batch(self, batch):
        inputs, targets, caps = batch

        inputs = np.transpose(inputs, (0,3,1,2))
        targets = np.transpose(targets, (0,3,1,2))

        return inputs, targets, caps

    def compute_and_plot_results(self, batch, title= '', subset=15):

        #print np.shape(batch), np.shape(batch)[0], np.shape(batch)[0][2]
        if np.shape(batch[0])[3]==3:
            self.test_and_plot(self.extract_batch(batch), title, subset)
        else :
            self.test_and_plot(batch, title, subset)







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
        self.input_var = input_var
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


        self.net = net[incoming_layer]


    def test_and_plot(self, batch, title, subset=15):

        full_title = 'Input, ground truth and generated image ' + title
        inputs, targets, caps = batch


        #inputs and targets already transposed for computation
        #must be retranspose for visualization
        if np.shape(inputs[0])==subset:
            indices = [i for i in range(subset)]
        else:
            indices = np.random.randint(inputs.shape[0], size=subset)

        plt.figure(dpi=subset*15)
        plt.title(full_title)
        for i in range(subset):

            idx = indices[i]

            fake_imgs = self.get_imgs(inputs[idx:idx+1])

            #True and generated center
            target_center = np.transpose(targets[idx], (1,2,0))
            fake_center = np.transpose(fake_imgs[0], (1,2,0))

            #Contour
            input_contour = np.transpose(inputs[idx],(1,2,0))

            #True image (with center)
            full_img=np.zeros((64,64,3))
            np.copyto(full_img,input_contour)
            full_img[16:48,16:48, :] = target_center

            #Fake image (with generated center)
            full_fake_img = np.zeros((64,64,3))
            np.copyto(full_fake_img, input_contour)
            full_fake_img[16:48,16:48, :] = fake_center


            plot_image = np.concatenate((input_contour, full_img,full_fake_img), axis=0)
            if i==0:
                all_images = plot_image
            else:
                all_images = np.concatenate((all_images, plot_image), axis = 1)

        plt.axis('off')
        plt.imshow(all_images)
        plt.show()
        #return net, incoming_layer

class AE_contour2center_captions(Model):
    def __init__(self):
	       pass

    def build_network(self, input_var, captions_var,
                      conv_before_pool=[2,2],
                      n_filters = 32,
                      code_size = 100,
                      filter_size = 3,
                      pool_factor = 2,
                      all_caps = True):

    	self.conv_before_pool = conv_before_pool
    	self.code_size = code_size
    	self.n_filters = n_filters
    	self.all_caps = all_caps

        self.input_var = input_var
        self.captions_var = captions_var

    	n_block = len(conv_before_pool)


        #caption_layer = InputLayer((None, 64 ), captions_var)

    	net={}

        #35629 words in worddict.pkl
        net['captions_input'] = InputLayer((None, 7576), captions_var)
        net['captions_code'] = DenseLayer(net['captions_input'], 100)


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

        net['code_and_captions']=ConcatLayer([net[incoming_layer], net['captions_code']])
        incoming_layer = 'code_and_captions'

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


        self.net = net[incoming_layer]
        #return net, incoming_layer



    def compile_theano_functions(self, learning_rate = 0.001, comp_train=True, comp_valid=True, comp_get = True):


        input_var = self.input_var#T.tensor4('input img bx3x64x64')
        target_var = T.tensor4('inpainting target')
        captions_var = self.captions_var#T.matrix('captions var')
        
        
        if comp_train:
            print "Defining and compiling train functions"

            pred_img = lasagne.layers.get_output(self.net )
            loss = self.get_loss(pred_img, target_var)
            params = lasagne.layers.get_all_params(self.net , trainable=True)
            updates = lasagne.updates.adam(loss, params, learning_rate = learning_rate)
            train_fn = theano.function([input_var,captions_var,target_var], loss, updates = updates,
                                        allow_input_downcast=True)
            self.train_fn = train_fn

        if comp_valid:
            print "Defining and compiling valid functions"

            valid_pred_imgs = lasagne.layers.get_output(self.net ,deterministic=True)
            valid_loss = self.get_loss(valid_pred_imgs, target_var)
            valid_fn = theano.function([input_var, captions_var, target_var], valid_loss, allow_input_downcast=True)
            self.valid_fn = valid_fn

        if comp_get:
            print 'Defining and compiling get_imgs function'

            self.get_imgs = theano.function([input_var,captions_var], lasagne.layers.get_output(self.net ,deterministic = True),
                              allow_input_downcast=True)
        print "Done"

    def extract_batch(self, batch):

        def one_hot_all_captions(caps, vocab_size = 7574):
            #print 'size caps', np.shape(caps)
            caps = np.array(caps)
            n_samples = np.shape(caps)[0]


            #caps_onehot = np.random.normal(loc=0.0, scale=0.001, size=(n_samples,vocab_size+2)).astype('float32')
            caps_onehot = np.zeros(shape =(n_samples, vocab_size+2), dtype=np.float32)
            
            for i in range(n_samples):

                caps_i_onehot = np.zeros(shape=(len(caps[i]), vocab_size+2 ), dtype=np.float32)

				if self.all_caps:
					for j in range(len(caps[i])):
	
						cap_j = caps[i][j]
						for word in cap_j:
							caps_i_onehot[j][word] = 1.0
	
					caps_onehot[i]=np.sum(caps_i_onehot, axis = 0)
					
				else:
					cap_i0 = caps[i][0]
					for word in cap_i0:
						caps_i_onehot[0][word] = 1.0
					caps_onehot[i] = np.array(caps_i_onehot[0])
            return caps_onehot

        inputs, targets, caps = batch

        inputs = np.transpose(inputs, (0,3,1,2))
        targets = np.transpose(targets, (0,3,1,2))
        caps_1hot =one_hot_all_captions(caps)

        return inputs, targets, caps_1hot


    def test_and_plot(self, batch, title, subset=15):

        full_title = 'Input, ground truth and generated image ' + title
        inputs, targets, caps = batch


        #inputs and targets already transposed for computation
        #must be retranspose for visualization
        if inputs.shape[0]==subset:
            indices = [i for i in range(subset)]
        else:
            indices = np.random.randint(inputs.shape[0], size=subset)

        plt.figure(dpi=subset*15)
        plt.title(full_title)
        for i in range(subset):

            idx = indices[i]

            fake_imgs = self.get_imgs(inputs[idx:idx+1], caps[idx:idx+1])

            #True and generated center
            target_center = np.transpose(targets[idx], (1,2,0))
            fake_center = np.transpose(fake_imgs[0], (1,2,0))

            #Contour
            input_contour = np.transpose(inputs[idx],(1,2,0))

            #True image (with center)
            full_img=np.zeros((64,64,3))
            np.copyto(full_img,input_contour)
            full_img[16:48,16:48, :] = target_center

            #Fake image (with generated center)
            full_fake_img = np.zeros((64,64,3))
            np.copyto(full_fake_img, input_contour)
            full_fake_img[16:48,16:48, :] = fake_center


            plot_image = np.concatenate((input_contour, full_img,full_fake_img), axis=0)
            if i==0:
                all_images = plot_image
            else:
                all_images = np.concatenate((all_images, plot_image), axis = 1)

        plt.axis('off')
        plt.imshow(all_images)
        plt.show()


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





        self.net = net[incoming_layer]
        return net, incoming_layer

if __name__ == '__main__':
    input_var = T.tensor4('input')
    captions_var = T.matrix('captions')
    ae, last_layer =AE_contour2center_captions().build_network(input_var, captions_var)
    layers = lasagne.layers.get_all_layers(ae[last_layer])
    #print layers
    layers2 = lasagne.layers.get_all_layers(ae)
    #print layers2
    for l in layers2 :
        print l, ae[l].output_shape