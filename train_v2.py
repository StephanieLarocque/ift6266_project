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
import models_v2 as models



def train(
        #Training hyper-parameters
        learning_rate = 0.001,
        weight_decay = 0,#TODO :NOT implemented yet!!
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
    G_input_var = T.tensor4('generator_input_var') #m noise samples
    D_input_var = T.tensor4('true images')

    gan = models.gan(G_input_var = G_input_var, D_input_var = D_input_var)


        # #Print layers and shape (to debug)
    # print 'Generator layers'
    # for layer in lasagne.layers.get_all_layers(generator):
    #     print layer, layer.output_shape
    # print 'Discriminator layers'
    # for layer in lasagne.layers.get_all_layers(discriminator):
    #     print layer, layer.output_shape

    G_predictions = lasagne.layers.get_output(gan.G['last_layer'])
    D_predictions = lasagne.layers.get_output(gan.D['last_layer'])
    D_over_G_predictions = lasagne.layers.get_output(gan.D_over_G['last_layer'])

    #####################################
    # Define and compile theano functions
    #####################################
    print "Defining and compiling theano functions"


    #Losses for each model
    #TODO : verifier que les loss sont bonnes
    D_loss = -(T.log(D_predictions) + T.log(1- D_over_G_predictions)).mean()
    G_loss = (T.log(1-D_over_G_predictions)).mean()#binary_crossentropy(D_predictions, D_target_var)

    # if weight_decay > 0:
    #     gen_weightsl2 = regularize_network_params(
    #         generator, lasagne.regularization.l2)
    #     discr_weightsl2 = regularize_network_params(
    #         discriminator, lasagne.regularization.l2)
    #     gen_loss += weight_decay * gen_weightsl2
    #     discr_loss += weight_decay * discr_weightsl2

    #Update parameters for each model

    D_params = lasagne.layers.get_all_params(gan.D['last_layer'], trainable=True)
    D_updates = lasagne.updates.adam(D_loss, D_params, learning_rate = learning_rate)
    #discr_acc = $dicriminator accuracy
    D_train_fn = theano.function([G_input_var,D_input_var], D_loss, updates = D_updates)

    G_params = lasagne.layers.get_all_params(gan.G['last_layer'], trainable=True)
    G_updates = lasagne.updates.adam(G_loss, G_params, learning_rate = learning_rate)
    #gen_acc = #generator accuracy
    G_train_fn = theano.function([G_input_var], G_loss, updates=G_updates)




    # ## Objectives
    # obj_d = T.mean(T.log(prediction_d) + T.log(1-prediction_dg))
    # obj_g = T.mean(T.log(prediction_dg))
    #
    # ## Updates
    # updates_d = lasagne.updates.momentum(1-obj_d, params_d, learning_rate = 0.01)
    # updates_g = lasagne.updates.momentum(1-obj_g, params_d_g, learning_rate = 0.01)
    # In [200]:
    # ## Train functions ##
    # train_d = theano.function([input_var_g, input_var_d], obj_d, updates=updates_d, allow_input_downcast=True)
    #
    # train_g = theano.function([input_var_g], obj_g, updates=updates_g, allow_input_downcast=True)
    # In [201]:
    # ## Output functions##
    # out_d = theano.function([input_var_d], prediction_d, allow_input_downcast=True)
    # out_dg = theano.function([input_var_g], prediction_dg, allow_input_downcast=True)
    # out_g = theano.function([input_var_g], prediction_g, allow_input_downcast=True)







if __name__=="__main__":
    train()
