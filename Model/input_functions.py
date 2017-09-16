# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:57:28 2017

@author: maxime
"""
import tensorflow as tf

# Mapping of tf.example features used in two places. One for the two.
features_spec = {"sequence": tf.VarLenFeature(tf.int64),
                 "sequence_length": tf.FixedLenFeature((), tf.int64, default_value=0),
                 "label": tf.FixedLenFeature((), tf.int64, default_value=0)}

def _parse_function(example_proto):
    """Function in charge of parsing a tf.example into a tensors"""
    # Parse the tf.example according to the features_spec definition
    parsed_features = tf.parse_single_example(example_proto, features_spec)
    sequence = parsed_features["sequence"]
    # Convert the sparse sequence tensor to dense.
    sequence_d = tf.sparse_to_dense(sequence.indices, sequence.dense_shape, sequence.values)
    # Return all the elements
    return parsed_features["sequence_length"], parsed_features["label"], sequence_d

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
    # Create an arbitrary bucket range.
    buckets = [tf.constant(num, dtype=tf.int64) for num in range(0, 500, 15)]
    # Number of elements per bucket.
    window_size = 1000
    # Group the dataset according to a bucket key (see bucketing_fn).
    # Every element in the dataset is attributed a key (here a bucket id)
    # The elements are then bucketed according to these keys. A group of 
    # `window_size` having the same keys are given to the reduc_fn. 
    dataset = dataset.group_by_window(
            lambda x, y, z: bucketing_fn(x, buckets), 
            lambda key, x: reduc_fn(key, x, window_size), window_size)
    # We now have buckets of `window_size` size, let's batch and pad them
    dataset = dataset.padded_batch(batch_size, padded_shapes=(
            tf.TensorShape([]), tf.TensorShape([]), tf.Dimension(None)))
    dataset = dataset.map(lambda x, y, z: 
        ({"sentence":z, "sequence_length":x}, y))
    # Repeat the dataset for a given number of epoch
    dataset = dataset.repeat(num_epochs)
    # Create the iterator to enumerate the elements of the dataset.
    iterator = dataset.make_one_shot_iterator()
    # Generator returned by the iterator.
    features, labels = iterator.get_next()
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

#def input_fn(seq_filename, label_filename, vocab, batch_size, num_epochs, skip, count):
#    """
#    Function to perform the data pipeline for the model.
#    Should be wrapped around an anonymous function to set the params
#    
#    Args:
#        seq_filename: a string with the path for the sequences to read
#        label_filename: a string with the path for the labels to read
#        vocab: a dict with characters as key and ids as values ({'a': 1, 'b':2, ...})
#        batch_size: the batch size to use
#        num_epochs: the number of times to read the entire dataset
#    """
#    
#    # Function to split a string into characters and encode them with their
#    # id from the vocab dict.
#    # /!\ Needed becasue the tf.string_split splits UTF-8 chars (ex: \xc3\xc8) into two characters.
#
#    #keys = list(vocab.keys())
#    mapping_strings = tf.constant(keys)
#    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings)
#    
#    filenames = [seq_filename] # Filename to look for the sentences (raw text file)
#    # Create a dataset out of the raw textfile
#    features_dataset = tf.contrib.data.TextLineDataset(filenames)
#    # Apply the split_string function to each row of the raw textfile. -> [tf.int32]
#    features_dataset = features_dataset.map(lambda string: tf.py_func(split_string, [string], tf.int32)) 
#    # Hopefully will be able to use this in the future.
#    features_dataset = features_dataset.map(lambda string: tf.string_split([tf.compat.as_str(string)], delimiter='').values)
#    features_dataset = features_dataset.map(lambda words: (table.lookup(words), tf.size(words)))
#    # Create a new dataset with the sequence length
#    features_dataset = features_dataset.map(lambda words: (words, tf.size(words)))
#    
#    # Same thing but for the labels
#    filenames = [label_filename]
#    # Create a dataset with the labels
#    labels_dataset = tf.contrib.data.TextLineDataset(filenames)
#    # Cast the labels from string type to int32 type
#    labels_dataset = labels_dataset.map(lambda string: tf.string_to_number(string, out_type=tf.int32))
#    
#    # Concatenate the two dataset together.
#    dataset = tf.contrib.data.Dataset.zip((features_dataset, labels_dataset))
#    # Batch and pad the data set: [sentence, sentence_length, label]
#    # Nothing to pad for sequence_length and label
#    # The sentences will be padded according to the largest sentence of the batch
#    dataset = dataset.padded_batch(batch_size, padded_shapes=(
#            (tf.Dimension(None), tf.TensorShape([])), tf.TensorShape([])))
#    # Reformat or the dataset as to make it easier to use in the model.
#    dataset = dataset.map(lambda words, labels: 
#        ({"sentence":words[0], "sequence_length":words[1]}, labels))
#    # Skip the first `skip` number of lines (val/train/test set param)
#    dataset = dataset.skip(skip)
#    # Only take `count` number of lines (val/train/test set param)
#    dataset = dataset.take(count)
#    # Repeat the dataset for a given number of epoch
#    dataset = dataset.repeat(num_epochs)
#    # Shuffle the dataset
#    dataset = dataset.shuffle(10*batch_size, seed=0)
#    # Create the iterator 
#    # -> Creates an Iterator for enumerating the elements of this dataset.
#    iterator = dataset.make_one_shot_iterator()
#    features, label = iterator.get_next()
#    return features, label