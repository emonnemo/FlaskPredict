from flask import Flask, redirect, url_for, request, render_template
from predict.prediction import predictResult
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/about')
def about():
   return render_template('about.html')

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/predict',methods = ['POST'])
def predict():
   if request.method == 'POST':
      age = request.form['age']
      workclass = request.form['workclass']
      fnlwgt = request.form['fnlwgt']
      education = request.form['education']
      educationNumber = request.form['educationNumber']
      maritalStatus = request.form['maritalStatus']
      occupation = request.form['occupation']
      relationship = request.form['relationship']
      race = request.form['race']
      sex = request.form['sex']
      capitalGain = request.form['capitalGain']
      capitalLoss = request.form['capitalLoss']
      hoursPerWeek = request.form['hoursPerWeek']
      nativeCountry = request.form['nativeCountry']
      
      return predictResult(model, age, workclass, fnlwgt, education, educationNumber,
			maritalStatus, occupation, relationship, race, sex,
			capitalGain, capitalLoss, hoursPerWeek, nativeCountry)

if __name__ == '__main__':
   model = joblib.load('static/model/model.pkl')
   app.run(debug = True)