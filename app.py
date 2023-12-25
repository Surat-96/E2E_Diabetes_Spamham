from flask import Flask,render_template,redirect,url_for,request
import pickle
import numpy as np

from flask_mysqldb import MySQL
import MySQLdb.cursors
import re


app = Flask(__name__)

#diabetes model read
filename = open('Diabetes/dpmod.pkl', 'rb')
model = pickle.load(filename)
filename.close()

#spamham model read
filename = open('Spam_Ham/spam-sms-mnb-model.pkl', 'rb')
classifier = pickle.load(filename)
files = open('Spam_Ham/cv-transform.pkl','rb')
cv = pickle.load(files)
filename.close()
files.close()


@app.route('/')
def index():
	return render_template('home.html')

@app.route('/home')
def home():
	return render_template('home.html')
    
@app.route('/diapredict', methods=['GET','POST'])
def diapredict():
    if request.method == 'POST':
        na = request.form['na']
        pr = int(request.form['pr'])
        gl = int(request.form['gl'])
        bp = int(request.form['bp'])
        st = int(request.form['st'])
        ins = int(request.form['in'])
        bm = float(request.form['bm'])
        dp = float(request.form['dp'])
        ag = int(request.form['ag'])
        
        data = np.array([[pr,gl,bp,st,ins,bm,dp,ag]])
        my_prediction = model.predict(data)
        my_prediction_proba = model.predict_proba(data)[0][1]
        
        return render_template('diashow.html',name=na,prediction=my_prediction,proba=my_prediction_proba)
    return render_template('dia.html')

@app.route('/sphampredict',methods=['GET','POST'])
def sphampredict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('sphamshow.html', prediction=my_prediction)
    return render_template('spham.html')

        
if __name__ == '__main__':
	app.run(debug=True)
