# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:40:12 2017

@author: maxime
"""
import tensorflow as tf
from cell import ConvLSTMCell

def cnnlstm(features, labels, mode, params):
    """
    Model to be used in the tf.estimator. Basically the machine learning model.
    Simple RNN model that:
        - Takes a sentence represented like ['This', 'is', 'a', 'sentence']
          where each character in a word is represented by a integer and each word
          in a batch has the same length (zero padded)
        - One word at a time, each word is embedded using a CNN and a Highway 
          network. (TODO: add the highway network)
        - This embedding is given to a RNN
        - The last state is given to another RNN (+ Attention over the previous
          hidden state) that predicts the next word.
    
    Args:
        - features: a dict: 
            - sequence: a tensor of shape [batch_size, max_sentence_length, max_word_size]
            filled with the character ids, and padded with 0
            - sequence_length: a tensor of shape [batch_size] with the original length of 
            the sequences.
            - max_word_size: tha maximum length of each word in the batch
        - labels: a dict:
            - sequence: a tensor of shape [batch_size, max_sentence_length] filled
            with the words ids of each sentence and padded with 0.
            - sequence_length: a tensor of shape [batch_size] with the original length of 
            the sequences.
        - mode: the mode of the model (given by the estimator)
        - params: a dict with the following keys:
            - vocab_size: the size of the character vocabulary used
            - embedding_size: the size of the embeddings
            - dropout: 1 - dropout probability (the keep probability)
    """
    ###########
    # ENCODER #
    ###########
    # Characters embeddings matrix. Basically each character id (int)
    # is associated a vector [char_embedding_size]
    embeddings_c = tf.Variable(tf.random_uniform([params['char_vocab_size'], 
                               params['char_embedding_size']], -1.0, 1.0))
    # Embed every char id into their embedding. Will go from this dimension
    # [batch_size, max_sequence_length, max_word_size] to this dimension
    # [batch_size, max_sequence_length, max_word_size, char_embedding_size]
    embedded_chars = tf.nn.embedding_lookup(embeddings_c, features['sequence'])
    print('embedded_chars',embedded_chars)
    # Do a convolution on the inputs
    pooled_outputs = []
    for i, window_size in enumerate(self._window_sizes):
        with tf.name_scope("conv-maxpool-%s" % self._num_filters):
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
            max_pooled = tf.reduce_max(conv, 1) 
            print('Max pooled', max_pooled)
#                pooled = tf.layers.max_pooling1d(
#                    conv,
#                    pool_size=[conv.get_shape()[1]],
#                    strides=filter_size,
#                    name="pool")
#                print('Pooled layer', pooled)
            # Append the result to the global results.
            pooled_outputs.append(max_pooled)
    # Create the actual encoder. Which applies a convolution on the char input
    # to have an embedding for each word. This embedding is then fed to the 
    # classical LSMT RNN.
    # TODO: apply dropout
    cell = ConvLSTMCell(num_units=100, window_sizes=[2, 3, 4], num_filters=20, 
                        embedding_size=params['char_embedding_size'])
    # Loop over the inputs and apply the previously created cell at every 
    # timestep. Returns the output at every step and last hidden state.
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, 
                                             inputs=embedded_chars, 
                                             sequence_length=features['sequence_length'])
    
    ###########
    # DECODER #
    ###########
    # Words embeddings matrix. Basically every word id (int) in the vocab
    # is associated a vector [char_embedding_size]
    embeddings_w = tf.Variable(tf.random_uniform([params['char_vocab_size'], 
                               params['word_embedding_size']], -1.0, 1.0))
    
    # Decoder cell. Basic LSTM cell that will do the decoding.
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=100)
    # Projection layer. Layer that takes the output of the decoder cell 
    # and projects it on the word vocab dimension.
    projection_layer = tf.layers.dense(params['word_vocab_size'], use_bias=False)
    # If not at infering mode, use the decoder_inputs
    # output at each time step.
    if mode != tf.estimator.ModeKeys.INFER:
        # Decoder outputs, i.e., what we are trying to predict.
        decoder_o = labels['sequence_output']
        # Embed the decoder input
        decoder_i = tf.nn.embedding_lookup(embeddings_w, labels['sequence_input'])
        # Helper method. Basically a function that "helps" the decoder
        # at each time step by giving it the true input, whatever it computed
        # earlier.
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_i, labels['sequence_length'])
    else:
        # Helper method. At inference time it is different, we do not have the
        # true inputs, so this function will take the previously generated output
        # and embbed it with the decoder embeddings.
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings_w, 
                            params['start_token'], params['end_token'])
    # The final decoder, with its cell, its intial state, its helper function,
    # and its projection layer. 
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, 
                                              encoder_state, 
                                              output_layer=projection_layer)
    # Use this decoder to perform a dynamic decode.
    # Dynamic Decoder: controls the flow of operations and mainly store the outputs
    # and keeps decoding until the decoder is done.
    # Decoder: kind of the cell of the dynacmic decoder. It passes the inputs
    # to the RNN, samples the output of the RNN and computes the next input.
    # To sample and compute the next inputs, the decoder uses a Helper function.
    # During training it is a TrainingHelper and during inference it is GreedyEmbeddingHelper
    # In our case the sampling is simply taking the argmax of the output logit.
    # The main difference between the two helpers is on the way they "compute" 
    # the next input. TrainingHelper will use the decoder inputs provided while 
    # the GreedyEmbeddingHelper will use the sampled RNN output and give it to
    # an embedding function to give it at as the next input. 
    # Outputs of the BasicDecoder is a BasicDecoderOutput which holds the logits
    # and the sample_ids.
    outputs, state, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)   
    # Contains the 
    logits = outputs.rnn_output # output of the projection layer
    sample_id = outputs.sample_id # argmax of the logits
    # If we are INFER time only
    if mode == tf.estimator.ModeKeys.INFER: 
        # Return a dict with the sample word ids.
        predictions = {"sequence": sample_id}
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, 
                                          export_outputs=export_outputs)
    
    # We are not at INFER time. We compute the cross entropy.
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_o, 
                                                              logits=logits)
    # Here we create a mask to "erase" the loss where the sentences are finished
    target_w = tf.sequence_mask(labels['sequence_length'], dtype=logits.dtype)
    # We apply the mask and sum the loss accross all the dimensions and divide it
    # by the batch size to make it independent of the batch_size.
    loss = (tf.reduce_sum(crossent * target_w) / tf.shape(decoder_i)[0])
    
    # At train time only.
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Initialize an optimize that has for goal to minimize the loss
        optimizer = tf.train.AdamOptimizer(0.001)
        # Apply gradient clipping
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), 
                                             global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    # Compute the accuracy of the model (the number of sequences that the model
    # got right)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=decoder_o, predictions=sample_id)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)