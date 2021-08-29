from os import listdir
from Feature_Extraction import extract_features
from pickle import dump


#feature = extract_features("img3")
#print(len(feature))

#dump(feature, open('features.pkl','wb'))
"""

weights="model.h5"

model = load_model(weights)

# load the tokenizer
tokenizer = load(open( 'tokenizer_resnet50.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 35

#xx="1434607942_da5432c28c.jpg"
#photo = query_feature('img_upload/'+xx)



testImagesLabel = load_2("static/img2")
train_descriptions =load_clean_description( 'description.txt', testImagesLabel)

#print(train_descriptions[xx[:-4]])


x=1
y=1725
rr=testImagesLabel[1725:]
for i in rr:

    photo = query_feature('static/img2/'+i+".jpg")

	
    print(y)
    y+=1

    predicted_description = generate_desc(model, tokenizer, photo, 39)

    yhat = predicted_description.split()
    #predicted.append(yhat)
    references = [d.split() for d in train_descriptions[i]]
    
    bleu_score_1 = corpus_bleu([references], [yhat] , weights=(1, 0, 0, 0) )
    bleu_score_2 = corpus_bleu([references], [yhat] , weights=(0.5, 0.5, 0, 0))
    bleu_score_3 = corpus_bleu([references], [yhat] , weights=(0.33, 0.33, 0.34, 0))
    bleu_score_4 = corpus_bleu([references], [yhat] , weights=(0.25, 0.25, 0.25, 0.25))
    bleu_score = ( 8*bleu_score_4 + 4*bleu_score_3 + 2*bleu_score_2 + bleu_score_1 )/15
    if bleu_score > 0.4:

        img= cv2.imread("static/img2/"+i+".jpg")
        cv2.imwrite("test/" +i+".jpg" , img)

    	
        print(x)
        x+=1

        
        print(bleu_score)  """
