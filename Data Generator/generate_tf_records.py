# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:46:37 2017

@author: maxime
"""

from mistake import Mistake
import io
import tensorflow as tf
from utils import create_example, get_vocab

i = 0
mistake_generator = Mistake()
vocab = get_vocab('data/vocab.json')

training_writer = tf.python_io.TFRecordWriter("data/training.tfrecord")
validation_writer = tf.python_io.TFRecordWriter("data/validation.tfrecord")
testing_writer = tf.python_io.TFRecordWriter("data/testing.tfrecord")

validation_size = 10000
validation_i = validation_size
testing_i = 2*validation_size

with io.open("data/europarl-v7.fr-en.fr", 'r', encoding='utf8') as fin:
    # For every line in the document.
    for line in fin.readlines():
        len_line = len(line)
        if (len_line < 10) | (len_line > 500):
            continue
        i += 1
        if i % 10000 == 0:
            print('Starting line number {i}'.format(i=str(i)))
        # First process the correct line.
        line = line.strip()
        example = create_example(line, 0, vocab)
        mistake_line = mistake_generator.generate_mistake(line.strip())
        counter_example = create_example(mistake_line, 1, vocab)
        if i <= validation_i:
            validation_writer.write(example.SerializeToString())
            validation_writer.write(counter_example.SerializeToString())
        elif i <= testing_i:
            testing_writer.write(example.SerializeToString())
            testing_writer.write(counter_example.SerializeToString())
        else:
            training_writer.write(example.SerializeToString())
            training_writer.write(counter_example.SerializeToString())
        if i % 100000 == 0:
            # break
            pass

validation_writer.close()
testing_writer.close()
training_writer.close()