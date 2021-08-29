from os import listdir
from pickle import dump
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import Model

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from numpy import array

from help_func import *

#extract features from all photos in library
def extract_features(directory):
    #load model
    model = ResNet50()
    
    #remove classification layer
    model.layers.pop()
    model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
    
    features = dict()
    
    #os.listdir --> returns a list of files in the directory
    for name in listdir(directory):
        filename = directory + '/' + name
        #load image
        image = load_img(filename, target_size = (224,224))
        #reshaping image into 4D for fitting in model
        image = img_to_array(image)
        image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        
        #The preprocess_input function is meant to adequate your image to the format the model requires
        image = preprocess_input(image)
        
        #extract features
        feature = model.predict(image, verbose = 0)
        #remove .jpg
        image_id = name.split('.')[0]
        features[image_id] = feature
        print(feature.shape)
    return features


def query_feature( img ):

    model = ResNet50()
    
    #remove classification layer
    model.layers.pop()
    model = Model(inputs = model.inputs, outputs = model.layers[-1].output)

    image = load_img(img, target_size = (224,224))
        #reshaping image into 4D for fitting in model
    image = img_to_array(image)
    image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        
        #The preprocess_input function is meant to adequate your image to the format the model requires
    image = preprocess_input(image)
        
        #extract features
    feature = model.predict(image, verbose = 0)

    return feature

def define_model(vocab_size, max_length):
    #feature extractor model
    inputs1  = Input(shape=(1000,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation = 'relu')(fe1)
    #Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    #decoder model
    decoder1 = add([fe2,se3])
    decoder2 = Dense(256, activation = 'relu')(decoder1)
    outputs = Dense(vocab_size, activation = 'softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    #print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model



# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length , vocab_size ):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo , vocab_size)
            yield [in_img, in_seq], out_word
                        

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo , vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)




"""
tokenizer = load(open( 'tokenizer2.pkl', 'rb'))

testImagesLabel = load_set("img3")
train_descriptions =load_clean_description( 'description.txt', testImagesLabel)
print(len(train_descriptions))

#tokenizer = create_tokenizer(train_descriptions)

#dump(tokenizer, open('tokenizer2.pkl', 'wb'))

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print(max_length)

train_features= load_photo_features("features.pkl" ,testImagesLabel )

model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 18
steps = len(train_descriptions)
for i in range(epochs):

    print(i+1)
    # create the data generator
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length ,vocab_size )
    # fit for one epoch
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1) 
    
    # save model

model.save('model_3.h5')   """