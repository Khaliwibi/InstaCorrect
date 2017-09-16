# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 08:58:44 2017

@author: maxime
"""

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import io
import numpy as np
import json
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

with io.open("vocab.json", 'r', encoding='utf8') as fin:
        vocab=json.loads(fin.readline())

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/api/is_correct', methods=['POST'])
# curl -H "Content-Type: application/json" -X POST -d '{"sentence":"this is my sentence"}' http://0.0.0.0/api/is_correct
def is_correct():
    data = request.get_json()
    text = data['sentence']
    # Cut the text into (possible) multiple sentences.
    print(text)
    sentences = sent_tokenize(text)
    correct_probability = 1
    if len(sentences) > 10 | len(text) > 1000:
        return "Please reduce the size of the text", 400
    for i, sentence in enumerate(sentences):
        tmp_response = correct_sentence(sentence)
        correct_probability = correct_probability * tmp_response['correct']
    response = {'correct': correct_probability, 'uncorrect': 1-correct_probability}
    return jsonify(response)

def encode_line(line, vocab):
    """Given a string and a vocab dict, encodes the given string"""
    line = line.strip()
    sequence = [vocab.get(char, -1) for char in line]
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

def correct_sentence(sentence):
    """Ask the server to know if the given sentence is correct"""
    host = "tfserver"
    port = 9000
    model_name = "instacorrect"
    example = create_example(sentence, 0, vocab).SerializeToString()
    serialized_examples = tf.contrib.util.make_tensor_proto([example])

    channel = implementations.insecure_channel(host, int(port))

    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs['examples'].CopyFrom(serialized_examples)
    result = stub.Predict(request, 5.0)  # 5 seconds
    float_vals = result.outputs['probabilities'].float_val
    response = {'correct': float_vals[0], 'incorrect': float_vals[1]}
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
