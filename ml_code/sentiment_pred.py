import pandas as pd
import re
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
from keras import backend as k
"""
txt = []
f = open('input.txt','r',encoding='utf8')
txt.append(f.read())
f.close()"""



def preprocess_text(sen):
    sentence = remove_tags(sen) 
    sentence = re.sub('[^a-zA-Z]', ' ', sentence) #psntuations and numbers remove
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence) #space removal
    sentence = re.sub(r'\s+', ' ', sentence) #remove blank spaces
    return sentence
	
TAG_RE = re.compile(r'<[^>]+>') #remove html tags
def remove_tags(text):
    return TAG_RE.sub('', text)
	

	
	
def predict_sentiment(txt):
	k.clear_session()
	df = pd.DataFrame([txt], columns =['text'])
	X = []
	sentences = list(df['text'])
	for sen in sentences:
		X.append(preprocess_text(sen))

	f = open('ml_code/tokenizer.pickle','rb')
	tokenizer = pickle.load(f)
	f.close()
	model = load_model('ml_code/model_final.model')


	X = tokenizer.texts_to_sequences(X)
	vocab_size = len(tokenizer.word_index) + 1
	maxlen = 100

	X_test = pad_sequences(X, padding='post', maxlen=maxlen)



	pred = model.predict(X_test)[0]

	out_index=np.argmax(pred)
	print(out_index)
	label=["negative","positive"]

	pred_class=label[out_index]


	#print("\n\n***********************************OUTPUT*********************************************\n\n")

	#print("Predicted Class of news is:",pred_class)
	return pred_class

"""
f = open('output.txt','w')
f.write(pred_class)
f.close()"""
#out=predict_sentiment("good movie")
#print(out)