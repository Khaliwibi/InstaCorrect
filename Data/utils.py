# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 08:55:51 2017

@author: maxime
"""

import tensorflow as tf
import io
import json
from nltk.tokenize import word_tokenize

def encode_line(line, vocab):
    """Given a string and a vocab dict, encodes the given string"""
    line = line.strip()
    sequence = [vocab.get(char, vocab['<UNK>']) for char in line]
    sequence_length = len(sequence)
    return sequence, sequence_length

def encode_line_charwise(line, vocab):
    """Given a string will encode it into the right tf.example format"""
    # Encode the string into their vocab representation
    splited = word_tokenize(line)
    sequence_length = len(splited)
    max_word_length = max([len(word) for word in splited])
    # Should have a one array of int.
    sequence = []
    pad_char = vocab.get('|PAD|')
    for word in splited:
        word_encoded = [vocab.get(char, vocab['|UNK|']) for char in word]
        word_encoded += [pad_char]*(max_word_length - len(word))
        sequence.extend(word_encoded)
    return sequence, sequence_length, max_word_length

def encode_line_wordwise(line, vocab):
    """Given a string and vocab, return the word encoded version"""
    splited = word_tokenize(line)
    sequence_input = [vocab.get(word, vocab['|UNK|']) for word in splited]
    sequence_input = [vocab['|GOO|']] + sequence_input
    sequence_output = sequence_input[1:] + [vocab['|EOS|']]
    sequence_length = len(sequence_input)
    return sequence_input, sequence_output, sequence_length

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_example(correct, mistake, word_vocab, char_vocab):
    """Given a string and a label (and a vocab dict), returns a tf.Example"""
    m_sequence, m_s_l, m_m_w_l = encode_line_charwise(mistake, char_vocab)
    c_sequence_i, c_sequence_o, c_s_l, = encode_line_wordwise(correct, word_vocab)
    example = tf.train.Example(features=tf.train.Features(feature={
            'correct_sequence_input': _int64_feature(c_sequence_i),
            'correct_sequence_output': _int64_feature(c_sequence_o),
            'correct_sequence_length': _int64_feature([c_s_l]),
            'mistake_sequence': _int64_feature(m_sequence),
            'mistake_sequence_length': _int64_feature([m_s_l]),
            'mistake_max_word_length':_int64_feature([m_m_w_l])}))
    return example

def get_vocab(filename):
    with io.open(filename, 'r', encoding='utf8') as fin:   
        vocab=json.loads(fin.readline())
    return vocab