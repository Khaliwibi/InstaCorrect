# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 08:55:51 2017

@author: maxime
"""

import tensorflow as tf
import io
import json

def encode_line(line, vocab):
    """Given a string and a vocab dict, encodes the given string"""
    line = line.strip()
    sequence = [vocab.get(char, vocab['<UNK>']) for char in line]
    sequence_length = len(sequence)
    return sequence, sequence_length

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_example(line, label, vocab):
    """Given a string and a label (and a vocab dict), returns a tf.Example"""
    sequence, sequence_length = encode_line(line, vocab)
    example = tf.train.Example(features=tf.train.Features(feature={
            'sequence': _int64_feature(sequence),
            'sequence_length': _int64_feature([sequence_length]),
            'label': _int64_feature([label])}))
    return example

def get_vocab(filename):
    with io.open(filename, 'r', encoding='utf8') as fin:   
        vocab=json.loads(fin.readline())
    return vocab