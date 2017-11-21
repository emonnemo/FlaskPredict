from flask import Flask, redirect, url_for, request, render_template
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin
import numpy
import pandas as pd
from sklearn import preprocessing

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == numpy.dtype('O') else X[c].mode() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def predictResult(model, age, workclass, fnlwgt, education, educationNumber,
	maritalStatus, occupation, relationship, race, sex,
	capitalGain, capitalLoss, hoursPerWeek, nativeCountry):
	input = [(age, workclass, fnlwgt, education, educationNumber,
		maritalStatus, occupation, relationship, race, sex,
		capitalGain, capitalLoss, hoursPerWeek, nativeCountry)]
	labels = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
		"occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
		"hours-per-week", "native-country"]

	input_data = pd.DataFrame.from_records(input, columns=labels)

	# drop columns
	# TODO: should make this not only fnlwgt, but depends later
	drop_list = ['fnlwgt']
	input_data.drop(drop_list, axis=1, inplace=True)

	# preprocessing the input
	# input_data = pd.DataFrame(input)
	input_data = DataFrameImputer().fit_transform(input_data)

	# categorize the string attributes to integer using label encoder used in train data
	le = preprocessing.LabelEncoder()
	le.classes_ = numpy.load('static/encoder/workclass.npy')
	input_data['workclass'] = le.transform(input_data['workclass'])

	le.classes_ = numpy.load('static/encoder/education.npy')
	input_data['education'] = le.transform(input_data['education'])

	le.classes_ = numpy.load('static/encoder/marital-status.npy')
	input_data['marital-status'] = le.transform(input_data['marital-status'])

	le.classes_ = numpy.load('static/encoder/occupation.npy')
	input_data['occupation'] = le.transform(input_data['occupation'])

	le.classes_ = numpy.load('static/encoder/relationship.npy')
	input_data['relationship'] = le.transform(input_data['relationship'])

	le.classes_ = numpy.load('static/encoder/race.npy')
	input_data['race'] = le.transform(input_data['race'])

	le.classes_ = numpy.load('static/encoder/sex.npy')
	input_data['sex'] = le.transform(input_data['sex'])

	le.classes_ = numpy.load('static/encoder/native-country.npy')
	input_data['native-country'] = le.transform(input_data['native-country'])

	prediction = model.predict(input_data)
	if prediction[0] == 0:
		return '<=50K'
	else:
		return '>50K'