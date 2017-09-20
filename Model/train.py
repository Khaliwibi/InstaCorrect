import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)
import os
import json
from input_functions import input_fn, serving_input_receiver_fn
from model import cnnlstm
import argparse
import io

def get_vocab(filename):
    with io.open(filename, 'r', encoding='utf8') as fin:   
        vocab=json.loads(fin.readline())
    return vocab

def train():
    """Perform the training of the model"""
    char_vocab = get_vocab('../Data/data/char_vocab_dict.json')
    word_vocab = get_vocab('../Data/data/words_vocab_dict.json')
    # Parameters given to the estimator. Mainly the size of the vocabulary
    # the embedding size to use and the (keep) drop out percentage
    model_params = {'char_vocab_size': len(char_vocab), 
                    'char_embedding_size': 30, 
                    'dropout': 0.75,
                    'word_vocab_size': len(word_vocab),
                    'word_embedding_size': 100,
                    'start_token': word_vocab['|GOO|'],
                    'end_token': word_vocab['|EOS|']}
    # The batch size to use for the train/valid/test set
    batch = 64
    # The number of times to train the model on the entire dataset    
    epochs = 10
    # The part of the dataset that will be skipped to be used by the training
    # and testing dataset
    # Lambda function used in the experiment. Returns a dataset iterator
    data_train = lambda: input_fn("../Data/data/training.tfrecord", batch, epochs)
    data_valid = lambda: input_fn("../Data/data/validation.tfrecord", batch, epochs)

    # Set the TF_CONFIG environment to local to avoid bugs
    os.environ['TF_CONFIG'] = json.dumps({'environment': 'local'})
    # Create a run config
    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=5*60, # Every 30 min
                                        log_device_placement=True,
                                        tf_random_seed=0,
                                        save_summary_steps=1000)
    # Create the estimator, the actual RNN model, with the defined directory
    # for storing files, the parameters, and the config.
    estimator = tf.estimator.Estimator(model_fn=cnnlstm, 
                                       model_dir="output/", 
                                       params=model_params, 
                                       config=config)
    # Give this estimator to an experiment to run the traning and validation,
    # input_functions, etc. From the tf documentation: "After an experiment is 
    # created (by passing an Estimator and inputs for training and evaluation),
    # an Experiment instance knows how to invoke training and eval loops [...]
    # eval_step = None so the evaluation step uses the entire validation set
    experiment = tf.contrib.learn.Experiment(estimator=estimator, 
                                             train_input_fn=data_train,
                                             eval_input_fn=data_valid, 
                                             eval_steps=None,
                                             local_eval_frequency=1, 
                                             min_eval_frequency=1)
    experiment.train_and_evaluate()
    
def inference():
    """Perform an inference on the latest checkpoint available"""
    """Perform the training of the model"""
    char_vocab = get_vocab('../Data/data/char_vocab_dict.json')
    word_vocab = get_vocab('../Data/data/words_vocab_dict.json')
    # Parameters given to the estimator. Mainly the size of the vocabulary
    # the embedding size to use and the (keep) drop out percentage
    model_params = {'char_vocab_size': len(char_vocab), 
                    'char_embedding_size': 30, 
                    'dropout': 0.75,
                    'word_vocab_size': len(word_vocab),
                    'word_embedding_size': 100,
                    'start_token': word_vocab['|GOO|'],
                    'end_token': word_vocab['|EOS|']}
    # The batch size to use for the train/valid/test set
    batch = 64
    # The number of times to train the model on the entire dataset    
    epochs = 10
    # The part of the dataset that will be skipped to be used by the training
    # and testing dataset
    # Lambda function used in the experiment. Returns a dataset iterator
    data_test = lambda: input_fn("../Data/data/testing.tfrecord", batch, epochs)

    # Set the TF_CONFIG environment to local to avoid bugs
    os.environ['TF_CONFIG'] = json.dumps({'environment': 'local'})
    # Create a run config
    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=60*30, # Every 30 min
                                        log_device_placement=True,
                                        tf_random_seed=0,
                                        save_summary_steps=1000)
    # Create the estimator, the actual RNN model, with the defined directory
    # for storing files, the parameters, and the config.
    estimator = tf.estimator.Estimator(model_fn=cnnlstm, 
                                       model_dir="output/", 
                                       params=model_params, 
                                       config=config)
    result = estimator.predict(data_test)
    return result

def export():
    """Export the last saved graph"""
    vocab = get_vocab('input/vocab.json')
    # Parameters given to the estimator. Mainly the size of the vocabulary
    # the embedding size to use and the (keep) drop out percentage
    model_params = {'vocab_size': len(vocab) + 1, 'embedding_size': 30, 
                    'dropout': 1}
    estimator = tf.estimator.Estimator(model_fn=cnnlstm, 
                                       model_dir="output/", 
                                       params=model_params)
    estimator.export_savedmodel("output/model_serving", serving_input_receiver_fn)
    
def input_inspection():
    """Inspect the inputs for inconsistency"""

    reverse_char_vocab = get_vocab('../Data/data/char_vocab_reve.json')
    char_vocab = get_vocab('../Data/data/char_vocab_dict.json')
    reverse_word_vocab = get_vocab('../Data/data/words_vocab_reve.json')
    
    features, labels = input_fn("../Data/data/validation.tfrecord", 5, 1)
    sess = tf.Session()
    feat, labl = sess.run([features, labels])
    sentences = feat['sequence']
    correct_input = labl['sequence_input']
    correct_output = labl['sequence_output']
    
    pad_char = char_vocab['|PAD|']
    for i in range(sentences.shape[0]):
        shape = sentences[i,:,:].shape
        sentence = sentences[i,:,:].tolist()
        for x in range(shape[0]):
            for y in range(shape[1]):
                if sentence[x][y] != pad_char:
                    print(reverse_char_vocab.get(str(sentence[x][y]), '<UNK>'), end='')
            print(' ', end='')
        print('END')
        print('****')
    
    for i in range(correct_input.shape[0]):
        ids = correct_input[i,:].tolist()
        words = [reverse_word_vocab.get(str(wid)) for wid in ids]
        print(" ".join(words))
        
        ids = correct_output[i,:].tolist()
        words = [reverse_word_vocab.get(str(wid)) for wid in ids]
        print(" ".join(words))

if __name__ == "__main__":
    
    # hey=inference(['Maxime'])
    # next(hey)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', dest='predict', action="store_true")
    parser.add_argument('--export', dest='export', action='store_true')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--feature', dest='feature', action='store_true')
    parser.add_argument('--to_predict', type=str, help='The str to pred.')
    FLAGS, unparsed = parser.parse_known_args()
    
    print(FLAGS)
    
    if FLAGS.predict:
        inference(FLAGS.to_predict)
    elif FLAGS.export:
        print('Start export')
        export()
    elif FLAGS.train:
        train()
    else:
        pass