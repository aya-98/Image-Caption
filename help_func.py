
import os 

from pickle import load

import pickle
from PIL import Image
from pickle import dump
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from numpy import argmax
from numpy import array
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import numpy as np


def load_doc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()    
    return text


def load_set(folder):

    data2=[]
    dataset = os.listdir(folder)
    for data in dataset:

        data2.append(data[:data.index('.')])

    return set(data2)

def load_2(folder):

    data2=[]
    dataset = os.listdir(folder)
    for data in dataset:

        data2.append(data[:data.index('.')])

    return data2






def load_clean_description(txtfile , dataset):


    doc = load_doc(txtfile)



    descriptions = dict()

    for l in doc.split('\n'):

        

        if len(l)<1 :

            continue

        		

        l=l.replace('"' , '')

        l=l.replace(',',  ' ' ,1)

        tokens=l.split()
        x=tokens[0]
        x=x[:x.index('.')]
        y=tokens[1:]
        if '.' in y :

            y.remove('.')

        if x in dataset:
            if x not in descriptions:
                descriptions[x] = list()
            desc = 'startseq ' + ' '.join(y) + ' endseq'
            descriptions[x].append(desc)


    return descriptions


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(des) for des in descriptions[key]]
    return all_desc




def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer    



def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)



def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



def generate_desc(model, tokenizer, photo, max_length):
    #start seq
    in_text = 'startseq'
    #repeatedly generate next letter using created in_text until it is not 'endseq' 
    for i in range(max_length):
        #convert to tokens
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        #padding
        sequence = pad_sequences([sequence], maxlen = max_length)
        #find probability
        yhat = model.predict([photo, sequence], verbose=0)
        #how?
        yhat = argmax(yhat)
        #map integer to word
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word 
        if word == "endseq":
            break
    return in_text


def load_photo_features(filename, dataset):
    # load all features
    all_features = pickle.load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


def extract( image_path):

    model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    fc2_model = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    img = image.load_img(image_path, target_size=fc2_model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    features = fc2_model.predict(x)[0]
    return features






