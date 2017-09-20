# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:43:18 2017

@author: maxime
"""

import tensorflow as tf

batch_size = None
time_steps = None
max_word_s = None
embed_spac = 30
# Counter
i = tf.constant(0)
# Loop while i < max_time_step
c = lambda i, x, z: tf.less(i, 3)
# Inputs
inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, 
                                                 time_steps, 
                                                 max_word_s, 
                                                 embed_spac])
# Mock output with zeros, should be erased afterwards
# dimension of output: [batch_size, time_steps, final_word_dim]
# We will concat the convolutions along the time_steps dimension
outputs = tf.placeholder(tf.float32, shape=[batch_size, 1, 10])

def body(i, inputs, outputs):
    # Varialbe to hold the diff convolutions over different 
    # windows.
    pooled_outputs = []
    # Perform a convolution on the inputs of the timestep i
    for window in [3,4,5]:
        conv = tf.layers.conv1d(inputs[:,i,:,:],
                         filters=10,
                         padding='same',
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

r = tf.while_loop(c, body, [i, inputs, outputs], 
      shape_invariants=[i.get_shape(), 
                        inputs.get_shape(), 
                        tf.TensorShape([None, None, None])])


conv = tf.layers.conv1d(inputs[:,i,:,:],
                     filters=15,
                     padding='same',
                     kernel_size=2)

max_pooled = tf.reduce_max(conv, 1) 