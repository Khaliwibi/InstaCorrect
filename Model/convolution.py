# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:43:57 2017

@author: maxime
"""

import tensorflow as tf

class Convolution(object):
    
    def __init__(self):#, window_sizes, num_filters, embedding_size):
        pass
#        self._window_sizes = window_sizes
#        self._num_filters = num_filters
#        self.embedding = embedding_size
        
    def body(self, i, inputs, outputs):
        # Varialbe to hold the diff convolutions over different 
        # windows.
        pooled_outputs = []
        # Perform a convolution on the inputs of the timestep i
        for window in [2,3,4]:
            conv = tf.layers.conv1d(inputs[:,i,:,:],
                             filters=50,
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_size=window)
            # The result is a tensor of [batch_size, max_word, num_filters]
            # To perform a max over time pooling, we take the max accross
            # the characters axis (axis 1)
            max_pooled = tf.reduce_max(conv, 1)
            # The result is a tensor of shape [batch_size, num_filters]
            # We can append it to the list containing all conv results
            pooled_outputs.append(max_pooled)
        # We now need to combine all of the results into one along the 
        # second dimension.
        all_outputs = tf.concat(pooled_outputs, 1)
        # To be able to concatenate it we expand the last dimension to
        # make it a 3D tensor.
        all_outputs = tf.expand_dims(all_outputs, 2) 
        # This step is contactenated with the previous steps
        outputs = tf.concat([outputs, all_outputs], 2)
        # Increment the counter with one.
        tf.add(i, 1)
        # Return the variables.
        return i, inputs, outputs


    def __while__(self, inputs):
        """ """
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        embed_size = 3*50
        # Looping variable
        i = tf.constant(0)
        # Loop while i < max_time_step
        c = lambda i, x, z: tf.less(i, 3)
        # The convolution will result in a tensor of shape 
        # [batch, max_time_step, numfilters*len(windo_sizes)]
        outputs = tf.zeros([batch_size, time_steps, embed_size], tf.float32)
        # Body -> the convolution function
        i, f_inputs,f_outputs = tf.while_loop(c, 
                                   self.body, 
                                   [i, inputs, outputs], 
                                   shape_invariants=[i.get_shape(), 
                                                     inputs.get_shape(), 
                                                     tf.TensorShape([None, None, None])])
        return outputs[:, 1:, :]
    
    def __call__(self, inputs):
        return self.__while__(inputs)