# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:43:57 2017

@author: maxime
"""

import tensorflow as tf

class Convolution(object):
    
    def __init__(self, window_sizes, num_filters, embedding_size):
        self._window_sizes = window_sizes
        self._num_filters = num_filters
        self.embedding = embedding_size
        
    def convolute(inputs, outputs, i, window_sizes, num_filters, s_embed):
        """Body function of the while loop"""
        # must take the inputs at time step i and convolute them
        pooled_outputs = []
        print('Inputs', inputs)
        # For each 
        for i, window_size in enumerate(window_sizes):
            with tf.name_scope("conv-maxpool-%s" % window_size):
                # Convolution Layer. 
                # Inputs are [batch_size, max_word_length, embedding_size]
                # Should apply a 1D convolution with padding = "SAME" to have
                # the same size as the input. Will return a tensor of shape
                # [batch_size, max_word_length, num_filters] on which we apply
                # a max over time pooling -> i.e. a reduce_max over the second
                # dimension.
                conv = tf.layers.conv1d(
                    inputs[:,i,:,:], # inputs at the ith time step
                    filters=num_filters, # the number of filters to apply
                    kernel_size=window_size, # the kernel size.
                    use_bias=True,
                    stride=1,
                    padding="SAME",
                    activation=tf.nn.relu,
                    name="conv-{i}".format(i=str(i)))
                print('Conv layer', conv)
                print('Input shape', inputs.get_shape())
                # Max-pooling over the outputs. Just take the maximum for each
                # filter along the max_word_length dimension. Will return a tensor
                # of shape [batch_size, num_filters] -> so identitical whatever
                # the word length. Clever Kim.
                max_pooled = tf.reduce_max(conv, 2) 
                print('Max pooled', max_pooled)
#                pooled = tf.layers.max_pooling1d(
#                    conv,
#                    pool_size=[conv.get_shape()[1]],
#                    strides=filter_size,
#                    name="pool")
#                print('Pooled layer', pooled)
                # Append the result to the global results.
                pooled_outputs.append(max_pooled)
        # Combine all the pooled features
        # We now have an array of tensors of the shape [batch, num_filters]
        # Concatenate them to have [batch, num_filters*len(window_size)]
        # To be sure to understand, we now have each word embedded as vector
        # of size num_filters*len(window_size)
        concatenated_filters = tf.concat(pooled_outputs, 1)
        # then add them to the outputs variable
        outputs.append(concatenated_filters)
        # increment i
        tf.add(i, 1)
        return inputs, outputs, i, window_sizes, num_filters

    def __while__(self, inputs):
        """ """
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        embed_size = self._num_filters*len(self._window_sizes)
        # Looping variable
        i = tf.constant(0)
        # Loop while i < max_time_step
        c = lambda i: tf.less(i, time_steps)
        # The convolution will result in a tensor of shape 
        # [batch, max_time_step, numfilters*len(windo_sizes)]
        outputs1 = tf.zeros([batch_size, time_steps, embed_size], tf.float32)
        outputs2 = []
        # Body -> the convolution function
        inp, out, i, w_sizes, n_filters = tf.while_loop(c, convolute,
                                            loop_vars=[inputs, outputs2, 
                                                       self._window_sizes, 
                                                       self._num_filters])
        concatenated_filters = tf.concat(out, 1)
    
    def __call__(self, inputs):
        """
        Performs the convolution of the inputs
        args:
            - inputs: a tensor of shape [batch_size, max_time_step, max_word_length, embedding_size]
            filled with embeddings.
        """
        # Array containing all the results of the convolution.
        pooled_outputs = []
        print('Inputs', inputs)
        # For each 
        for i, window_size in enumerate(self._window_sizes):
            with tf.name_scope("conv-maxpool-%s" % self.window_size):
                # Convolution Layer. 
                # Inputs are [batch_size, max_word_length, embedding_size]
                # Should apply a 1D convolution with padding = "SAME" to have
                # the same size as the input. Will return a tensor of shape
                # [batch_size, max_word_length, num_filters] on which we apply
                # a max over time pooling -> i.e. a reduce_max over the second
                # dimension.
                conv = tf.layers.conv1d(
                    inputs,
                    filters=self._num_filters, # the number of filters to apply
                    kernel_size=[window_size], # the kernel size.
                    use_bias=True,
                    padding="SAME",
                    activation=tf.nn.relu,
                    name="conv-{i}".format(i=str(i)))
                print('Conv layer', conv)
                print('Input shape', inputs.get_shape())
                # Max-pooling over the outputs. Just take the maximum for each
                # filter along the max_word_length dimension. Will return a tensor
                # of shape [batch_size, num_filters] -> so identitical whatever
                # the word length. Clever Kim.
                max_pooled = tf.reduce_max(conv, 2) 
                print('Max pooled', max_pooled)
#                pooled = tf.layers.max_pooling1d(
#                    conv,
#                    pool_size=[conv.get_shape()[1]],
#                    strides=filter_size,
#                    name="pool")
#                print('Pooled layer', pooled)
                # Append the result to the global results.
                pooled_outputs.append(max_pooled)
        # Combine all the pooled features
        # We now have an array of tensors of the shape [batch, num_filters]
        # Concatenate them to have [batch, num_filters*len(window_size)]
        # To be sure to understand, we now have each word embedded as vector
        # of size num_filters*len(window_size)
        concatenated_filters = tf.concat(pooled_outputs, 1)
        print('concaneted_filters', concatenated_filters)
        #num_filters_total = self._num_filters * len(self._filter_sizes)
        #☺print('h_pool', h_pool)
        #h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        #print('h_pool_flat', h_pool_flat)
        return concatenated_filters