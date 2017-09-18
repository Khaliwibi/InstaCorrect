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
    outputs, state, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)   
    logits = outputs.rnn_output
    sample_id = outputs.sample_id
    
    if mode != tf.estimator.ModeKeys.INFER:
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=decoder_o, logits=logits)
    
    target_w = tf.sequence_mask(labels['sequence_length'], dtype=logits.dtype)
    train_loss = (tf.reduce_sum(crossent * target_w) / batch_size)
    
    
    
    
    
    
    
    # Output
    batch_range = tf.range(tf.shape(outputs)[0])
    indices = tf.stack([tf.cast(batch_range, tf.int64), features['sequence_length'] - 1], axis=1)
    last_output = tf.gather_nd(outputs, indices)

    # Final Layer
    logits = tf.layers.dense(inputs=last_output, units=2)
    softmax = tf.nn.softmax(logits)

    # Predictions
    classes = tf.argmax(input=softmax, axis=1)
    predictions = {"classes": classes, "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Summary
    #correct_prediction = tf.equal(tf.cast(labels, tf.int32), tf.cast(classes, tf.int32))
    correct_prediction = tf.equal(labels, classes)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.merge_all()

    # Loss Minimization
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(0.001)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
