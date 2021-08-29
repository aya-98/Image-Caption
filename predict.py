

from pickle import load
from keras.models import load_model
from  Feature_Extraction import query_feature
from help_func import *
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
#from keras.preprocessing import image
#from keras.applications.imagenet_utils import decode_predictions, preprocess_input
#from keras.models import Model
from flask import Flask, request, render_template , url_for

from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from werkzeug.utils import secure_filename

app = Flask(__name__)


UPLOAD_FOLDER = 'static/img_upload/'
 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

weights="model_3.h5"

model = load_model(weights)

# load the tokenizer
tokenizer = load(open( 'tokenizer2.pkl', 'rb'))
testImagesLabel = load_2("static/img3")
train_descriptions =load_clean_description( 'description.txt', testImagesLabel)



@app.route('/start')

def start():

	return render_template('home.html')









@app.route('/upload', methods=['POST'])

def upload():

    score=[]
    matched=[]

    if str(request.form['q']) =="":

        
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        photo = query_feature('static/img_upload/'+filename)

        predicted_description = generate_desc(model, tokenizer, photo, 35)
    
        print_description = ' '.join(predicted_description.split(' ')[1:-1])

        yhat = predicted_description.split()
    #yhat = (train_descriptions[filename[:-4]][0]).split()

        for i in testImagesLabel:


            references = [d.split() for d in train_descriptions[i]]
    
            bleu_score_1 = corpus_bleu([references], [yhat] , weights=(1, 0, 0, 0) )
            bleu_score_2 = corpus_bleu([references], [yhat] , weights=(0.5, 0.5, 0, 0))
            bleu_score_3 = corpus_bleu([references], [yhat] , weights=(0.33, 0.33, 0.34, 0))
            bleu_score_4 = corpus_bleu([references], [yhat] , weights=(0.25, 0.25, 0.25, 0.25))
            bleu_score = ( 8*bleu_score_4 + 4*bleu_score_3 + 2*bleu_score_2 + bleu_score_1 )/15
            if bleu_score > 0.45 and i != filename[:-4]:

                matched.append("img3/"+i+".jpg")
                score.append(bleu_score)


    
        z = [matched for _,matched in sorted(zip(score,matched))]

        z.reverse()



        if len(z) >=15:

            matched=z[:15]

            return render_template('display.html', filename='img_upload/'+filename , scores= matched , desc= print_description , l=15 , query=True)

    
        else:

            return render_template('display.html', filename='img_upload/'+filename , scores= z , desc= print_description ,l=len(z), query=True)


    
    elif str(request.form['q']) !="":


        des=str(request.form['q'])

        desc= "startseq "+des+ ' endseq'

        yhat = desc.split()
    #yhat = (train_descriptions[filename[:-4]][0]).split()

        for i in testImagesLabel:


            references = [d.split() for d in train_descriptions[i]]
    
            bleu_score_1 = corpus_bleu([references], [yhat] , weights=(1, 0, 0, 0) )
            bleu_score_2 = corpus_bleu([references], [yhat] , weights=(0.5, 0.5, 0, 0))
            bleu_score_3 = corpus_bleu([references], [yhat] , weights=(0.33, 0.33, 0.34, 0))
            bleu_score_4 = corpus_bleu([references], [yhat] , weights=(0.25, 0.25, 0.25, 0.25))
            bleu_score = ( 8*bleu_score_4 + 4*bleu_score_3 + 2*bleu_score_2 + bleu_score_1 )/15
            if bleu_score > 0.45 :

                matched.append("img3/"+i+".jpg")
                score.append(bleu_score)


    
        z = [matched for _,matched in sorted(zip(score,matched))]

        z.reverse()

        if len(z) >=15:

            matched=z[:15]

            return render_template('display.html', query=False , scores= matched , des=des  , l=15)

    
        else:

            return render_template('display.html', query=False  , scores= z , des=des ,l=len(z))  







    




    
          
   












    















# generate description
"""

predicted_description = generate_desc(model, tokenizer, photo, 39)
print_description = ' '.join(predicted_description.split(' ')[1:-1])

print(print_description  )





matchedFiles = list()

for img in testImagesLabel:
    if len(matchedFiles) > 20:
        break
    #actual, predicted = list(), list()
    
    yhat = predicted_description.split()
    #predicted.append(yhat)
    references = [d.split() for d in train_descriptions[img]]
    #x=(train_descriptions[img][0]).split()
    #y.append(x)
    #actual.append(references) 
    #print([references],[ yhat])
    bleu_score_1 = corpus_bleu([references], [yhat] , weights=(1, 0, 0, 0) )
    bleu_score_2 = corpus_bleu([references], [yhat] , weights=(0.5, 0.5, 0, 0))
    bleu_score_3 = corpus_bleu([references], [yhat] , weights=(0.33, 0.33, 0.34, 0))
    bleu_score_4 = corpus_bleu([references], [yhat] , weights=(0.25, 0.25, 0.25, 0.25))
    bleu_score = ( 8*bleu_score_4 + 4*bleu_score_3 + 2*bleu_score_2 + bleu_score_1 )/15
    if bleu_score > 0.5:
        matchedFiles.append(img)
        print(bleu_score)
        



#subplot(r,c) provide the no. of rows and columns

print(len(matchedFiles) )


#for file in matchedFiles:


#process("img2/"+file +".jpg")





for i in range(len(matchedFiles)):

    xy = cv2.imread("img2/" + matchedFiles[i]+".jpg")

    cv2.imshow("hello" , xy)

    cv2.waitKey(10000) """
    


