import io
import numpy as np
from nltk.tokenize import word_tokenize
import tensorflow as tf
from scipy.sparse import coo_matrix

class DataGenerator(object):
    def __init__(self):
        self.metadata = np.load('Inputs/metadata.npy').item()

    def __call__(self):
        pass

    def next(self):
        filename = 'Inputs/sentences.txt'
        sentences = []
        labels = []
        with io.open(filename, 'r', encoding='utf8') as fin:
            for i, line in enumerate(fin.readlines()):
                sentence, num_words = encode_line(line, self.metadata["char_to_int"], self.metadata["max_sentence_length"], self.metadata["max_word_length"])
                label = 1 if i % 2 == 0 else 0
                sentences.append((sentence, num_words))
                labels.append(label)
                if (i % 10) == 0:
                    yield (sentences, labels)
                    sentence = []
                    labels = []


def build_vocab(filenames):
    """
    Given a filename, builds the character vocabulary.
    """
    vocab = set()
    max_word_length = 0
    max_sentence_length = 0
    number_of_sentences = 0
    for filename in filenames:
        with io.open(filename, 'r', encoding='utf8') as fin:
            for line in fin.readlines():
                number_of_sentences += 1
                vocab = vocab | set(line)
                sentence_length = len(line)
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length
                if number_of_sentences % 1000 == 0:
                    print(str(number_of_sentences))
    vocab = list(vocab)
    char_to_int = {char:(i+1) for i, char in enumerate(vocab)}
    int_to_char = {(i+1):char for i, char in enumerate(vocab)}
    metadata = {"char_to_int": char_to_int,
                "int_to_char": int_to_char,
                "max_sentence_length": max_sentence_length,
                "number_of_sentences": number_of_sentences}
    return metadata

def build_tensors(filename, vocab, max_sentence_length, number_of_sentences):
    # Should get the file, iterate over it.
    # Should replace the chars with their ids
    # Should pad data
    tensor = np.zeros([number_of_sentences, max_sentence_length], dtype=np.int32)
    sequence_length = np.zeros([number_of_sentences], dtype=np.int32)
    with io.open(filename, 'r', encoding='utf8') as fin:
         for i, line in enumerate(fin.readlines()):
             line = line[:max_sentence_length]
             ids = [vocab.get(char, -1) for char in line]
             tensor[i,:len(ids)] = ids
             sequence_length[i] = len(ids)
             if i % 1000 == 0:
                 print(str(i))
    return tensor, sequence_length

def _featurize_py_func(text):
    """
    Given a string of text, returns the char encoded matrix [number_of_words, number_of_chars]
    """
    label = np.array(text[-1], dtype=np.int32)
    words = word_tokenize(text[:-2])
    chars = np.zeros([max_sentence_length, max_word_length], dtype=np.int32)
    for i, word in enumerate(words):
        ids = [char_to_int.get(char, -1) for char in word]
        chars[i,:len(ids)] = ids
    return chars


if __name__ == "__main__":
    metadata = build_vocab(['Inputs/sentences.txt'])
    np.save('Inputs/metadata.npy', metadata)
    metadata = np.load('Inputs/metadata.npy').item()
    features, sequence_length = build_tensors('Inputs/sentences.txt', metadata["char_to_int"], metadata["max_sentence_length"], metadata["number_of_sentences"])
    #labels = np.loadtxt("Inputs/labels.txt", dtype=np.int32)
    labels = np.empty(metadata["number_of_sentences"], dtype=np.int32)
    labels[::2] = 0
    labels[1::2] = 1
    np.save('Inputs/features.npy', features)
    np.save('Inputs/sequence_length.npy', sequence_length)
    np.save('Inputs/labels.npy', labels)
