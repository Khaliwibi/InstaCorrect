# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:57:28 2017

@author: maxime
"""
import tensorflow as tf

# Mapping of tf.example features used in two places. One for the two.
spec = {"correct_sequence_input": tf.VarLenFeature(tf.int64),
        "correct_sequence_output": tf.VarLenFeature(tf.int64),
        "correct_sequence_length": tf.FixedLenFeature((), tf.int64, default_value=0),
        "mistake_sequence": tf.VarLenFeature(tf.int64),
        "mistake_sequence_length": tf.FixedLenFeature((), tf.int64, default_value=0),
        "mistake_max_word_length": tf.FixedLenFeature((), tf.int64, default_value=0)}

def _parse_function(example_proto):
    """Function in charge of parsing a tf.example into a tensors"""
    # Parse the tf.example according to the features_spec definition
    parsed_features = tf.parse_single_example(example_proto, spec)
    # Sparse tensor 
    c_sequence_input_sparse = parsed_features['correct_sequence_input']
    # tensor with all the correct sentence encoded with ids [245, 245, ...]
    c_sequence_input_dense = tf.sparse_to_dense(c_sequence_input_sparse.indices, 
                               c_sequence_input_sparse.dense_shape, 
                               c_sequence_input_sparse.values)
    # Sparse tensor 
    c_sequence_output_sparse = parsed_features['correct_sequence_output']
    # tensor with all the correct sentence encoded with ids [245, 245, ...]
    c_sequence_output_dense = tf.sparse_to_dense(c_sequence_output_sparse.indices, 
                               c_sequence_output_sparse.dense_shape, 
                               c_sequence_output_sparse.values)
    correct_sequence_length = parsed_features['correct_sequence_length']
    mistake_sequence_sparse = parsed_features['mistake_sequence']
    mistake_sequence_dense = tf.sparse_to_dense(mistake_sequence_sparse.indices, 
                               mistake_sequence_sparse.dense_shape, 
                               mistake_sequence_sparse.values)
    mistake_sequence_length = parsed_features['mistake_sequence_length']
    mistake_max_word_length = parsed_features['mistake_max_word_length']
    # Tensor with the words encoded at the character level. To be able to add
    # it to a tf.example (which accepts only 1D list), the words are all the 
    # same dimension and should be reshaped once converted to dense.
    # the max_word_length is used for that.
    # [1, 0, 0, 1, 1, 0, 1, 1, 1] => [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    # This will result in words as rows, should be as column.
    mistake_sequence_dense = tf.reshape(mistake_sequence_dense, 
                                    tf.stack([
                                        tf.cast(mistake_sequence_length, tf.int32),
                                        tf.cast(mistake_max_word_length, tf.int32)
                                        ]))
    # Let's tranpose it to have it as columns.
    # mistake_sequence_dense = tf.transpose(mistake_sequence_dense)
    # Return all the elements
    to_return = (c_sequence_input_dense, c_sequence_output_dense, 
                 correct_sequence_length, mistake_sequence_dense, 
                 mistake_sequence_length, mistake_max_word_length)    
    return to_return

def bucketing_fn(sequence_length, buckets):
    """Given a sequence_length returns a bucket id"""
    t = tf.clip_by_value(buckets, 0, sequence_length)
    return tf.argmax(t)

def reduc_fn(key, elements, window_size):
    """Receives `window_size` elements"""
    return elements.shuffle(window_size, seed=0)

def input_fn(filenames, batch_size, num_epochs):
    """
    Function to perform the data pipeline for the model.
    Should be wrapped around an anonymous function to set the parameters.
    Args:
        seq_filename: a string with the path for the tf.records to read
        batch_size: the batch size to use
        num_epochs: the number of times to read the entire dataset
    """    
    # Create a dataset out of the raw TFRecord file. See the Data Generator for more
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    # Map the tf.example to tensor using the _parse_function
    dataset = dataset.map(_parse_function, num_threads=4)
    # Repeat the dataset for a given number of epoch
    dataset = dataset.repeat(num_epochs)
    # Create an arbitrary bucket range.
    buckets = [tf.constant(num, dtype=tf.int64) for num in range(0, 100, 5)]
    # Number of elements per bucket.
    window_size = 1000
    # Group the dataset according to a bucket key (see bucketing_fn).
    # Every element in the dataset is attributed a key (here a bucket id)
    # The elements are then bucketed according to these keys. A group of 
    # `window_size` having the same keys are given to the reduc_fn. 
    dataset = dataset.group_by_window(
            lambda a,b,c,d,m_s_l,e: bucketing_fn(m_s_l, buckets), 
            lambda key, x: reduc_fn(key, x, window_size), window_size)
    # We now have buckets of `window_size` size, let's batch and pad them
    dataset = dataset.padded_batch(batch_size, padded_shapes=(
        tf.Dimension(None), # Correct sentence input -> pad along only dimension
        tf.Dimension(None), # Correct sentence output -> pad along only dimension
        tf.TensorShape([]), # Correct sentence length
        (tf.Dimension(None), tf.Dimension(None)), # Mistake sent -> pad along both axis
        tf.TensorShape([]),# Mistake sentence length
        tf.TensorShape([])) # Mistake sentence max word length
    )
    # Let's now make it a bit more easy to understand this dataset by mapping
    # each feature.
    dataset = dataset.map(lambda a, b, c, d, e, f: 
        ({"sequence_input": a, "sequence_output": b, "sequence_length": c}, 
         {"sequence": d, "sequence_length": e, "max_word_length": f}))
    # Create the iterator to enumerate the elements of the dataset.
    iterator = dataset.make_one_shot_iterator()
    # Generator returned by the iterator.
    labels, features = iterator.get_next()
    return features, labels


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    # Placeholder for the tf.example to be received
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None])
    # Dict to be passed to ServingInputReceiver -> input signature
    receiver_tensors = {'examples': serialized_tf_example}
    # Parse the example to a dict of features
    features = tf.parse_example(serialized_tf_example, features_spec)
    # Take the sequence sparse tensore
    sequence = features['sequence']
    # Convert it to a dense representation
    sequence_dense = tf.sparse_to_dense(sequence.indices, sequence.dense_shape, 
                                        sequence.values)
    # The sequence length
    sequence_length = features['sequence_length']
    # The dict of features passed to the ServingInputReceiver
    features = {'sentence': sequence_dense, 'sequence_length': sequence_length}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)