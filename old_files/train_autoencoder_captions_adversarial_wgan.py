
# coding: utf-8

# In[1]:

#!/usr/bin/env python2
import os
import sys
import argparse
import time
from distutils.dir_util import copy_tree

import numpy as np
import random
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params
from lasagne.objectives import binary_crossentropy
from lasagne.nonlinearities import tanh

from iterator import Iterator
import autoencoder_model as ae_model
import models_v4 as gan_model

import PIL.Image as Image
from matplotlib import pyplot as plt
from matplotlib import colors


_FLOATX = config.floatX


# In[2]:

#Training hyper-parameters

learning_rate_gen = 0.01
learning_rate_discr = 0.0005
weight_decay = 0
num_epochs = 500
max_patience = 100
#data_augmentation={}
SAVEPATH = 'save_models'
LOADPATH = 'load_models'
batch_size = 400
extract_center = True
load_caption = True
#nb_discriminator_steps = 2

#Model Hyperparameters
conv_before_pool=[2,2]
n_filters = 32
code_size = 500
filter_size = 3
pool_factor = 2
clip_treshold = 0.05

all_caps = True
output_nonlinD = tanh

lambda_rec = 1.0
lambda_adv = 1.0

subset_train = 50
subset_valid = 10



# In[3]:

#######################
# Saving path and stuff
######################

exp_name = 'AEwgan_'+ ('5' if all_caps else '1') + 'caps'
#exp_name += '_nonlin='+str(output_nonlinD)
exp_name += '_lrG='+str(learning_rate_gen)
exp_name += '_lrD='+str(learning_rate_discr)
exp_name += '_lambREC=' +str(lambda_rec)
exp_name += '_lambADV='+str(lambda_adv)
exp_name += '_clip='+str(clip_treshold)
exp_name += '_wd='+str(weight_decay)
exp_name += '_bs='+str(batch_size)
exp_name += '_conv='+str(conv_before_pool)
exp_name += '_nfilt='+str(n_filters)
exp_name += '_code='+str(code_size)
exp_name += ('_subTRAIN'+str(subset_train)) if subset_train>0 else ''
exp_name += ('_subVALID'+str(subset_valid)) if subset_valid>0 else ''




savepath=os.path.join(sys.path[1],SAVEPATH, exp_name)
loadpath=os.path.join(sys.path[1],LOADPATH, exp_name)

if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    print('\033[93m The following folder already exists {}. '
          'It will be overwritten in a few seconds...\033[0m'.format(
              savepath))
print 'Saving directory : ' + savepath
print 'Loading directory : '+ loadpath


# In[4]:

#######################
#Build dataset iterator
#######################

print "Loading training data..." #threads???
train_iter = Iterator(which_set='train', batch_size = batch_size,
            extract_center = extract_center, load_caption = load_caption)

print "Loading validation data..." #threads???
valid_iter = Iterator(which_set='valid', batch_size = batch_size,
            extract_center = extract_center, load_caption = load_caption)

test_iter = None

vocab_size = len(train_iter.vocab)
print 'Vocabulary size : ', vocab_size

n_batches_train = train_iter.n_batches
n_batches_valid = valid_iter.n_batches
n_batches_test = test_iter.n_batches if test_iter is not None else 0

print "Batch. train: %d, val %d, test %d" % (n_batches_train,
                            n_batches_valid, n_batches_test)


# In[5]:

# def put_center(contour, center):
#     n_samples = np.shape(contour)[0]
#     print np.shape(contour[0]), n_samples
#     full_img=np.zeros(contour)
#     np.copyto(full_img,contour)
#     full_img[:,:,16:48,16:48] = center

#     return full_img




# In[6]:

##########################################
# Build generator and discriminator models
##########################################
print('Defining symbolic variables and building models')

#Input and target var for the generator
ae_input_var = T.tensor4('input img bx3x32x32')
ae_captions_var = T.matrix('captions var')


model = ae_model.AE_contour2center_captions()

model.build_network(input_var = ae_input_var,
                  captions_var = ae_captions_var,
                  conv_before_pool=conv_before_pool,
                  n_filters = n_filters,
                  code_size = code_size,
                  all_caps = all_caps,
                  filter_size = filter_size,
                  output_nonlin = output_nonlinD,
                  pool_factor = pool_factor)


# In[7]:


ae_target_var = T.tensor4('target var bx3x32x32')

D = gan_model.discriminator()
D.build_network(input_var = ae_target_var, contour_var = ae_input_var, conv_before_pool = [1,1,1,1])

D_over_G = gan_model.discriminator_over_generator()
D_over_G.build_network(G_net = model, D_net=D)


model.compile_theano_functions(learning_rate= learning_rate_gen, comp_train = False, comp_valid=False)



#####################################
# Define and compile theano functions
#####################################

print "Defining and compiling theano functions"

#l1 = 0.5
#l2 = 1-l1

pred_img = lasagne.layers.get_output(model.net)
rec_loss = model.get_loss(pred_img, ae_target_var)


D_predictions = lasagne.layers.get_output(D.net)
D_over_G_predictions = lasagne.layers.get_output(D_over_G.net)


D_loss = -0.5*T.mean(D_predictions - D_over_G_predictions)


model_loss = lambda_rec * rec_loss - lambda_adv*T.mean(D_over_G_predictions)


print 'model.train_fn'
ae_params = lasagne.layers.get_all_params(model.net, trainable=True)
ae_updates = lasagne.updates.adam(model_loss, ae_params, learning_rate = learning_rate_gen)
model.train_fn = theano.function([ae_input_var,ae_captions_var, ae_target_var], model_loss, updates = ae_updates,
                            allow_input_downcast=True)



print 'D.train_fn'
D_params = lasagne.layers.get_all_params(D.net, trainable=True)
D_updates = lasagne.updates.adam(D_loss, D_params, learning_rate=learning_rate_discr)
D.train_fn = theano.function([ae_input_var, ae_captions_var, ae_target_var], D_loss, updates=D_updates,
                            allow_input_downcast=True, on_unused_input='ignore')





# In[44]:
print 'model.valid_fn'

valid_pred_imgs = lasagne.layers.get_output(model.net,deterministic=True)
valid_loss = model.get_loss(valid_pred_imgs, ae_target_var) - 0.5*T.mean(lasagne.layers.get_output(D_over_G.net))

valid_fn = theano.function([ae_input_var,ae_captions_var, ae_target_var], valid_loss, allow_input_downcast=True)
model.valid_fn = valid_fn

print 'Done'


# In[10]:

# D.get_pred = theano.function([ae_input_var, ae_captions_var, ae_target_var], D_predictions, on_unused_input='ignore')
# D_over_G.get_pred = theano.function([ae_input_var, ae_captions_var, ae_target_var], D_over_G_predictions, on_unused_input='ignore')
#
#
# # In[11]:
#
# D.get_before_nonlin = theano.function([ae_input_var, ae_captions_var, ae_target_var],
#                                       lasagne.layers.get_output(D.dict_net['last_layer']), on_unused_input='ignore')
# D_over_G.get_before_nonlin = theano.function([ae_input_var, ae_captions_var, ae_target_var],
#                                       lasagne.layers.get_output(D_over_G.dict_net['last_layer']), on_unused_input='ignore')


# In[111]:




# In[123]:

plot_results_train = False
plot_results_valid = False




# In[126]:

# Initialization

err_train = []
err_D_train = []
err_valid = []

best_err_val = 0
patience = 0

reset_best_results = True


# In[ ]:

#
# Train loop
#



# Training main loop
print "Start training"

for epoch in range(num_epochs):
    #learn_step.set_value((learn_step.get_value()*0.99).astype(theano.config.floatX))

    start_time = time.time()
    cost_train_epoch = 0
    cost_D_epoch = 0

    # Train

    for i, train_batch in enumerate(train_iter):

        if subset_train > 0 and i>= subset_train:
            break

        train_batch = model.extract_batch(train_batch)
        inputs_train, targets_train, caps_train = train_batch




        cost_D_batch = D.train_fn(inputs_train, caps_train, targets_train)
        cost_D_epoch += cost_D_batch
        #print 'D cost batch', i, cost_D_batch



        all_params = lasagne.layers.get_all_param_values(D.net)
        n_params = len(all_params)
        new_params = [[] for i in range(n_params)]
        for i in range(n_params):
            new_params[i] = np.array(all_params[i].clip(-clip_treshold, clip_treshold))

        lasagne.layers.set_all_param_values(D.net, new_params)



        cost_train_batch = model.train_fn(inputs_train, caps_train, targets_train)
        cost_train_epoch += cost_train_batch
        #print 'ae cost batch ' , cost_train_batch



    #Add epoch results
    err_train += [cost_train_epoch/n_batches_train]

    err_D_train += [cost_D_epoch/n_batches_train]



    # Validation
    cost_val_epoch = 0


    for i, valid_batch in enumerate(valid_iter):

        if subset_valid > 0 and i> subset_valid:
            break
        #rint 'valid', i

        valid_batch = model.extract_batch(valid_batch)
        inputs_valid, targets_valid, caps_valid = valid_batch

        # Validation step
        cost_val_batch = model.valid_fn(inputs_valid, caps_valid, targets_valid)
        #print i, 'validation batch cost : ', cost_val_batch


        #Update epoch results
        cost_val_epoch += cost_val_batch

    #Add epoch results
    err_valid += [cost_val_epoch/n_batches_valid]


    #Print results (once per epoch)
    out_str = "EPOCH %i: Avg cost train %f, cost discr train %f, cost val %f, took %f s"
    out_str = out_str % (epoch, err_train[epoch],
                         err_D_train[epoch],
                         err_valid[epoch],
                         time.time()-start_time)
    print out_str


    with open(os.path.join(savepath, "ae_output.log"), "a") as f:
        f.write(out_str + "\n")

    if epoch == 0 and reset_best_results:
        best_err_valid = err_valid[epoch]

        np.savez(os.path.join(savepath, 'ae_model_best.npz'),*lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath , "ae_errors_best.npz"), err_train=err_train, err_valid=err_valid)
        np.savez(os.path.join(savepath, 'ae_model_last.npz'), *lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath , "ae_errors_last.npz"), err_train=err_train, err_valid=err_valid)
        np.savez(os.path.join(savepath, 'D_model_best.npz'),*lasagne.layers.get_all_param_values(D.net))
        np.savez(os.path.join(savepath, 'D_model_last.npz'), *lasagne.layers.get_all_param_values(D.net))

    elif epoch > 1 and err_valid[epoch] < best_err_valid:
        print('saving best (and last) model')
        best_err_valid = err_valid[epoch]
        patience = 0
        np.savez(os.path.join(savepath, 'ae_model_best.npz'),*lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath , "ae_errors_best.npz"), err_train=err_train, err_valid=err_valid)
        np.savez(os.path.join(savepath, 'ae_model_last.npz'), *lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath , "ae_errors_last.npz"), err_train=err_train, err_valid=err_valid)

        np.savez(os.path.join(savepath, 'D_model_best.npz'),*lasagne.layers.get_all_param_values(D.net))
        np.savez(os.path.join(savepath, 'D_model_last.npz'), *lasagne.layers.get_all_param_values(D.net))

    else:
        patience += 1
        print('saving last model')
        np.savez(os.path.join(savepath, 'ae_model_last.npz'), *lasagne.layers.get_all_param_values(model.net))
        np.savez(os.path.join(savepath, 'D_model_last.npz'), *lasagne.layers.get_all_param_values(D.net))
        np.savez(os.path.join(savepath , "ae_errors_last.npz"), err_train=err_train, err_valid=err_valid)

    # Finish training if patience has expired or max nber of epochs reached
    if patience == max_patience or epoch == num_epochs-1:
        if savepath != loadpath:
            print('Copying model and other training files to {}'.format(loadpath))
            copy_tree(savepath, loadpath)
        break
