from flask import Flask,render_template,request,redirect,url_for,session
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords
import numpy as np
from ml_code import sentiment_pred
app = Flask(__name__)
DIR = os.getcwd()
app.config['UPLOAD_FOLDER']=DIR+"\\static\\media"
app.secret_key = "s3cr3t_k3y_for_my_flask_app"


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/index")
def index():
    return render_template('index.html')


 
@app.route('/review', methods=['GET', 'POST'])
def review():
    sentiment = None 
    if request.method == 'POST':
        review = request.form['review']
        print(f"Review: {review}")
        
        sentiment = sentiment_pred.predict_sentiment(review)
        print(f"Sentiment: {sentiment}")

    return render_template('index.html', sentiment=sentiment)

if __name__ == "__main__":
  app.run(debug=True)
