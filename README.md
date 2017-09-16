# InstaCorrect
A to Z implementation of a **deep learning** (french) spell checker:
* Data gathering and pre-processing
* Model definition
* Training and inference
* Model serving
* Front-end interaction

## Introduction
The ultimate goal of the project is to have a model that can effectively correct any french sentence for spell and grammatical mistakes. As this is probably a bit too much to ask for a first end-to-end model, the first step will be to classify bewteen erroneous and correct sentences.

## Data Gathering and Pre-Processing
#### Dataset
The first step is to find a relevant dataset. In this case, there are no pre-made dataset available. We will have to make our own! For that, I download all of the french translations from the (*European Parliament Proceedings Parallel Corpus 1996-2011*)[http://www.statmt.org/europarl/]. The french part contains around 2,190,579 sentences and 54,202,850 words. 
This gives us a set of correct sentences, at least in theory. We now have to come up with a set of erroneous sentences. We will generate them from correct one. In /Data Generator/mistake.py, you will find a Class that can generate a correct sentence into an erroneous one.
For example, it could map this correct sentence *Rien ne sert de courir, il faut partir à point* into *Rien ne sers de courir, il faut partir a point*. This mistake generator is still basic but at least we have somehting to start with.
Before creating our dataset, we will create a `vocabulary` or mapping of characters to intgers. For example, the letter `b` will be assigned the number `1`. Every character in our dataset will be assigned a number.
Now we can create our dataset. It will consist of `TFRecord` file filled with `TFExamples`. It is basically a fancy way to describe a list of dict. Each dicitionary will consist of:
1. The encoded sentence, i.e., an array of integer. Each interger representing the associated character from the `dictionary`. 
2. The sentence length, i.e, the number of characters in this sentence
3. The label, i.e, is this sentence correct or not.
This dataset will be split up in three parts: traning, validation and testing.

## Model Definition
Now that we have our dataset, we can define what our model will look like. For the moment, it is still basic as the goal was the have somehting running quickly. It will be further improved.
1. An embedding layer. 
2. A two layer RNN. 
3. A dense layer
