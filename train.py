#!/usr/bin/env python2

# Peut etre une facon de contourner le probleme :
#https://github.com/MayankSingal/Generative-Adversarial-Networks-Lasagne/blob/master/Generative%20Adversarial%20Network.ipynb

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
from lasagne.objectives import binary_crossentropy

from iterator import Iterator
import models



def train(
        #Training hyper-parameters
        learning_rate = 0.001,
        weight_decay = 0,#1e-4,
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
    # print "Loading training data..." #threads???
    # train_iter = Iterator(which_set='train', batch_size = batch_size,
    #             extract_center = extract_center, load_caption = load_caption)
    # print "Loading validation data..." #threads???
    # valid_iter = Iterator(which_set='valid', batch_size = batch_size,
    #             extract_center = extract_center, load_caption = load_caption)
    # test_iter = None
    #
    # n_batches_train = train_iter.n_batches
    # n_batches_valid = valid_iter.n_batches
    # n_batches_test = test_iter.n_batches if test_iter is not None else 0
    #
    # print "Batch. train: %d, val %d, test %d" % (n_batches_train,
    #                             n_batches_valid, n_batches_test)


    ##########################################
    # Define symbolic variables and
    # Build generator and discriminator models
    ##########################################
    print('Defining symbolic variables and building models')

    #Input and target var for the generator
    gen_input_var = T.tensor4('generator_input_var') #m noise samples
    gen_target_var = T.ivector('generator_target_var (all zeros)') #must be batch-size 'zeros'

    generator = models.generator().build_network(gen_input_var)
    fake_img = lasagne.layers.get_output(generator)[0]

    #And then for the discriminator
    true_img = T.tensor4('true images')
    #fake = T.tensor4()
    concatenation = theano.function([fake_img,true_img], T.concatenate([fake_img,true_img]))
    discr_input_var = concatenation(fake_img,true_img)
    #G(gen_input_var) and m real samples
    discr_target_var = T.ivector('discriminator_target_var (half zeros, half 1)')
    #m '0' and m '1' (if there is really m real samples)

    discriminator = models.discriminator().build_network(discr_input_var)
    discr_predictions = lasagne.layers.get_output(discriminator)[0] #for fake AND real data

    # #Print layers and shape (to debug)
    # print 'Generator layers'
    # for layer in lasagne.layers.get_all_layers(generator):
    #     print layer, layer.output_shape
    # print 'Discriminator layers'
    # for layer in lasagne.layers.get_all_layers(discriminator):
    #     print layer, layer.output_shape

    #####################################
    # Define and compile theano functions
    #####################################
    print "Defining and compiling theano functions"
    #Losses for each model
    gen_loss = binary_crossentropy(discr_predictions, gen_target_var)
    gen_loss = gen_loss.mean()
    discr_loss = binary_crossentropy(discr_predictions, discr_target_var)
    discr_loss = discr_loss.mean()

    # if weight_decay > 0:
    #     gen_weightsl2 = regularize_network_params(
    #         generator, lasagne.regularization.l2)
    #     discr_weightsl2 = regularize_network_params(
    #         discriminator, lasagne.regularization.l2)
    #     gen_loss += weight_decay * gen_weightsl2
    #     discr_loss += weight_decay * discr_weightsl2

    #Update parameters for each model
    gen_params = lasagne.layers.get_all_params(generator, trainable=True)
    gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate = learning_rate)
    #gen_acc = #generator accuracy
    gen_train_fn = theano.function([gen_input_var, gen_target_var], gen_loss, updates=gen_updates)

    discr_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    discr_updates = lasagne.updates.adam(discr_loss, discr_params, learning_rate = learning_rate)
    #discr_acc = $dicriminator accuracy
    discr_train_fn = theano.function([discr_input_var, discr_target_var], discr_loss, updates = discr_updates)








if __name__=="__main__":
    train()
