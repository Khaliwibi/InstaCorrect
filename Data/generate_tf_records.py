# -*- coding: utf-8 -*-
"""
The goal of this file is to generate the training examples for the model.
"""

from mistake import Mistake
import io
import tensorflow as tf
from utils import create_example, get_vocab
import random

i = 0
random.seed(0)
mistake_generator = Mistake()
char_vocab = get_vocab('data/char_vocab_dict.json')
word_vocab = get_vocab('data/words_vocab_dict.json')

training_writer = tf.python_io.TFRecordWriter("data/training.tfrecord")
validation_writer = tf.python_io.TFRecordWriter("data/validation.tfrecord")
testing_writer = tf.python_io.TFRecordWriter("data/testing.tfrecord") 

with io.open("data/europarl-v7.fr-en.fr", 'r', encoding='utf8') as fin:
    # For every line in the document.
    for line in fin.readlines():
        line = line.strip()
        correct_sentence_length = len(line)
        # If the sentence is too short or too long, discard it.
        if (correct_sentence_length < 10) | (correct_sentence_length > 500):
            continue
        # First process the correct line charwise
        correct_line = line.lower()
        mistake_line = mistake_generator.generate_mistake(line)
        # Make a tf.example out of the two lines
        example = create_example(correct_line, mistake_line, word_vocab, char_vocab)
        # Write the tf.example to a tf.record file
        random_number = random.random()
        if random_number <= 0.025:
            validation_writer.write(example.SerializeToString())
        elif random_number <= 0.05:
            testing_writer.write(example.SerializeToString())
        else:
            training_writer.write(example.SerializeToString())
        i += 1
        if i % 10000 == 0:
            print('Starting line number {i}'.format(i=str(i)))

validation_writer.close()
testing_writer.close()
training_writer.close()