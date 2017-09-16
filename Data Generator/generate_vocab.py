# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 08:38:54 2017

@author: maxime
"""

import io
import json

line_counter = 0
vocab = set()
vocab.add('<PAD>')
vocab.add('<UNK>')
vocab.add('<EOS>')
with io.open("data/europarl-v7.fr-en.fr", 'r', encoding='utf8') as fin:
    for line in fin.readlines():
        line_counter += 1
        vocab = vocab | set(line)
        if line_counter % 100000 == 0:
            print(str(line_counter))
    vocab = list(vocab)
    vocab.insert(0, '|UNK|')
    vocab.insert(0, '|EOS|')
    vocab.insert(0, '|PAD|')
    char_to_int = {char:(i) for i, char in enumerate(vocab)}
    int_to_char = {(i):char for i, char in enumerate(vocab)}

with io.open("data/vocab.json", 'w', encoding='utf8') as fin:   
    fin.write(json.dumps(char_to_int))
    
with io.open("data/reverse_vocab.json", 'w', encoding='utf8') as fin:   
    fin.write(json.dumps(int_to_char))