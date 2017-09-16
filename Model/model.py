# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:40:12 2017

@author: maxime
"""
import tensorflow as tf

def cnnlstm(features, labels, mode, params):
    """
    Model to be used in the tf.estimator. Basically the machine learning model.
    Simple RNN model that:
        - Embbeds the characters in a sentence
        - Give the embeddings to a two layers RNN
        - Add a softmax final layer on top of the last output of the RNN
        - Predict if the sentence is correct or not based on a given label.
    
    Args:
        - features: a dict containing two keys: 
            - x: a tensor of shape [batch_size, max_sentence_length_in_batch]
            padded with a given value.
            - sl: a tensor fo shape [batch_size] with the original length of 
            the sequences.
        - labels: a tensor of shape [batch_size] with 0 for correct sentences and
        1 for uncorrect sentences
        - mode: the mode of the model (given by the estimator)
        - params: a dict with the following keys:
            - vocab_size: the size of the vocabulary used
            - embedding_size: the size of the embeddings
            - dropout: 1 - dropout probability (the keep probability)
    """
    # Embeddings
    with tf.name_scope("embedding"):
        W = tf.Variable(tf.random_uniform([params['vocab_size'], params['embedding_size']], -1.0, 1.0), name="W")
        embedded_chars = tf.nn.embedding_lookup(W, features['sentence'])

    # RNN
    cell_one = tf.contrib.rnn.LSTMCell(num_units=100)
    cell_one = tf.contrib.rnn.DropoutWrapper(cell_one, output_keep_prob=params['dropout'])
    cell_two = tf.contrib.rnn.LSTMCell(num_units=100)
    cell_two = tf.contrib.rnn.DropoutWrapper(cell_two, output_keep_prob=params['dropout'])
    cells = tf.contrib.rnn.MultiRNNCell([cell_one, cell_two])
    outputs, last_states = tf.nn.dynamic_rnn(cell=cells, dtype=tf.float32, inputs=embedded_chars, sequence_length=features['sequence_length'])

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
