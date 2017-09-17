# InstaCorrect
End-to-end implementation of a **deep learning** (french) spell checker: www.instacorrect.com
* Data gathering and pre-processing
* Model definition
* Training and inference
* Model serving
* Front-end interaction

## Introduction
The ultimate goal of the project is to have a model that can effectively correct any french sentence for spell and grammatical mistakes. As this is probably a bit too much to ask for a first end-to-end model, the first step will be to classify bewteen erroneous and correct sentences.

## Data Gathering and Pre-Processing
The first step is to find a relevant dataset. In this case, there are no pre-made dataset available. We will have to make our own! For that, I download all of the french translations from the [*European Parliament Proceedings Parallel Corpus 1996-2011*](http://www.statmt.org/europarl/). The french part contains around 2,190,579 sentences and 54,202,850 words. 
This gives us a set of correct sentences, at least in theory. We now have to come up with a set of erroneous sentences. We will generate them from correct one. In /Data Generator/mistake.py, you will find a Class that can generate a correct sentence into an erroneous one.
For example, it could map this correct sentence *Rien ne sert de courir, il faut partir à point* into *Rien ne sers de courir, il faut partir a point*. This mistake generator is still basic but at least we have somehting to start with.
Before creating our dataset, we will create a `vocabulary` or mapping of characters to intgers. For example, the letter `b` will be assigned the number `1`. Every character in our dataset will be assigned a number.
Now we can create our dataset. It will consist of `TFRecord` file filled with `TFExamples`. It is basically a fancy way to describe a list of dict. Each dicitionary will consist of:
1. The encoded sentence, i.e., an array of integer. Each interger representing the associated character from the `dictionary`. 
2. The sentence length, i.e, the number of characters in this sentence
3. The label, i.e, is this sentence correct or not.
This dataset will be split up in three parts: traning, validation and testing.
Furthermore, there is a dyanmic padding and bucketing of the examples as to optimize the training time.

## Model Definition
Now that we have our dataset, we can define what our model will look like. For the moment, it is still basic as the goal was the have somehting running quickly. It will be further improved.
1. An embedding layer. 
2. A two layer LSTMs newtork with dropout. 
3. A dense layer

The loss is defined as the softmax cross entropy. It is minimized with the AdamOptimizer and gradient clipping. The model is built using tensorflow.

## Training and Inference
The training and inference is done using a tensorflow estimator. It is really useful and allows to focus on the essential part of the model and not the usual plumbing associated with running a tensorflow model. Furthermore, it is really easy to export a trained model using an estimator.

## Model Serving
Once a model is exported. It can be served to the "real world" with tensorflow serving. Tensorflow serving is not the easiest thing to grasp. According to me, it is not well documented. Tensorflow serving is a program that enables to put your model on a server. This server accepts gRPC requests and returns the output of the model. In theory it sounds easy, in practice it not that easy. 

Google released a binary that you can download using apt-get. This makes it much more easier. You just download this binary and execute. It will launch your server. This server expects to find exported models in the directory you specified. In this directory you simply copy-paste your saved model from earlier. That's it. You can customize it more, but it does the job for me.

Now that's great. But it's not really user-friendly to ask your uses to make gRPC request to your server. That's why you also need a simple application that can link the two together. Here I implemented a simple Flask application, that displays a simple website and exposes a single API function `is_correct`. When you make a POST request to this end-point with a text, it splits your text into sentences and calls the tensorflow-serving server with each sentences. It then multiplies the correct probabilities together to have a global correctnes probability and finally returns it to the front-end application.

All these application are bundle together using Docker Compose and live on a DigitalOcean (mini) droplet. 

## Front-end Application
A basic front-end application that runs angularjs to take the text written and send an AJAX request to the Flask app. 

## Results and conclusion
After one epoch (~4M examples), the model achieves an accuracy of 95% on the validation set. This is already great! But in practice the model fails to detect basic errors. This is probably due to two things:
* The mistake generator is not really good. It generates unlikely mistakes more than likely mistakes. Your model is only as good as the quality of your data.
* The model does not know about a *subset* of french, i.e., the translation of the european parlaments talks. Not really representative of your every chit chat...

How to improve the model ? Well the goal of this first version was to create a simple model but end-to-end. The next model will be a "predictive" one, it will try to correct your sentence instead of just saying whether it is correct or not. I will try to implement the following paper: [Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction](https://arxiv.org/abs/1604.04677). I will also try to improve the mistake generator. 

## Contributions
Feel free to contribute/comment/share this repo :-)
