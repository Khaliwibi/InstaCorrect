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
    vocab = get_vocab('input/vocab.json')
    # Parameters given to the estimator. Mainly the size of the vocabulary
    # the embedding size to use and the (keep) drop out percentage
    model_params = {'vocab_size': len(vocab) + 1, 'embedding_size': 30, 
                    'dropout': 0.75}
    # The batch size to use for the train/valid/test set
    batch = 64
    # The number of times to train the model on the entire dataset    
    epochs = 1
    # The part of the dataset that will be skipped to be used by the training
    # and testing dataset
    # Lambda function used in the experiment. Returns a dataset iterator
    data_train = lambda: input_fn("input/training.tfrecord", batch, epochs)
    data_valid = lambda: input_fn("input/validation.tfrecord", batch, epochs)

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
    
def inference(string):
    """Perform an inference on the latest checkpoint available"""
    # Parameters given to the estimator. Mainly the size of the vocabulary
    # the embedding size to use and the (keep) drop out percentage
    vocab = get_vocab('input/vocab.json')
    model_params = {'vocab_size': len(vocab) + 1, 'embedding_size': 30, 
                    'dropout': 1}
    sentence = [vocab.get(char, -1) for char in list(string)]
    sentence = np.array(sentence).reshape([1, len(sentence)])
    sentence_length = np.array(len(sentence)).reshape([1,])
    
    data_inference = tf.estimator.inputs.numpy_input_fn(
        x={"sentence": sentence, "sequence_length":sentence_length},
        num_epochs=1, shuffle=False)
    estimator = tf.estimator.Estimator(model_fn=cnnlstm, 
                                       model_dir="output/", 
                                       params=model_params)
    result = estimator.predict(data_inference)
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
    reverse_vocab = get_vocab('input/reverse_vocab.json')
    features, labels = input_fn("input/validation.tfrecord", 5, 1)
    sess = tf.Session()
    feat, labl = sess.run([features, labels])
    sentences = feat['sentence']
    sequence_length = feat['sequence_length']
    for i in range(sentences.shape[0]):
        sentence = sentences[i,:]
        sl = sequence_length[i]
        label = labl[i]
        sentence_str = "".join([reverse_vocab.get(str(num), "<UNK>") for num in sentence])
        print('**** \n input: {str} \n sequence length: {sl} \n label: {label}'.
              format(str=sentence_str, sl=str(sl), label=str(label)))

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