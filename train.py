#!/usr/bin/env python2

import os
import sys
import argparse
import time
#from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params

from iterator import Iterator
import models



def train(
        #Training hyper-parameters
        learning_rate = 0.001,
        weight_decay = 1e-4,
        num_epochs = 500,
        max_patience = 100,
        #data_augmentation={},
        savepath = 'save_models', #might need to change a bit
        loadpath = 'load_models',
        batch_size = 128,
        extract_center = True,
        load_caption = True,
        #Model Hyperparameters
        conv_before_pool = [1,1,1,1],
        n_filters = 64,
        filter_size = 3,
        n_units_dense_layer = 1024,
        out_nonlin = lasagne.nonlinearities.sigmoid,):

    ##############
    # Saving stuff
    ##############
    exp_name = 'gan'

    savepath=os.path.join(sys.path[0],savepath, exp_name)
    loadpath=os.path.join(sys.path[0],loadpath, exp_name)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    else:
        print('\033[93m The following folder already exists {}. '
              'It will be overwritten in a few seconds...\033[0m'.format(
                  savepath))
    print 'Saving directory : ' + savepath
    print 'Loading directory : '+ loadpath



    ########################
    # Build dataset iterator
    ########################
    print "Loading training data..." #threads???
    train_iter = Iterator(which_set='train', batch_size = batch_size,
                extract_center = extract_center, load_caption = load_caption)
    print "Loading validation data..." #threads???
    valid_iter = Iterator(which_set='valid', batch_size = batch_size,
                extract_center = extract_center, load_caption = load_caption)
    test_iter = None


    ############################
    # Define symbolic variables
    ############################

    gen_input_var = T.tensor4('generator_input_var')
    discr_input_var = T.tensor4('discriminator_input_var')
    #disc_target_var = T.ivector('discriminator_target_var') #TODO

    n_batches_train = train_iter.n_batches
    n_batches_valid = valid_iter.n_batches
    n_batches_test = test_iter.n_batches if test_iter is not None else 0

    print "Batch. train: %d, val %d, test %d" % (n_batches_train,
                                n_batches_valid, n_batches_test)

    ##########################################
    # Build generator and discriminator models
    ##########################################

    generator = models.generator().build_network(gen_input_var)
    discriminator = models.discriminator().build_network(discr_input_var)

    #Print layers and shape (to debug)
    print 'Generator layers'
    for layer in lasagne.layers.get_all_layers(generator):
        print layer, layer.output_shape
    print 'Discriminator layers'
    for layer in lasagne.layers.get_all_layers(discriminator):
        print layer, layer.output_shape

    #####################################
    # Define and compile theano functions
    #####################################

    print "Defining and compiling theano functions"

    fake_img = lasagne.layers.get_output(generator)[0]

    # gen_loss = 
    # disc_loss =








if __name__=="__main__":
    train()
