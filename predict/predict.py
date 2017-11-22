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
from sklearn import preprocessing
import numpy
import pandas as pd
import pickle

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

def oneHotEncode(listName, columnName, minRange, maxRange):
	# get column value
	columnValue = listName[columnName][0]

	# dropping this column and also concatenate the list with one hot columns
	listName = listName.drop(columnName, axis=1)
	for i in range(minRange, maxRange + 1):
		if columnValue == i:
			listName[columnName + str(i)] = 1
		else:
			listName[columnName + str(i)] = 0
	return listName

def labelEncode(listName, columnName, modeDict):
	# categorize the string attributes to integer using label encoder used in train data
	labelEncoder = preprocessing.LabelEncoder()
	labelEncoder.classes_ = numpy.load('static/encoder/{}.npy'.format(columnName))
	try:
		listName[columnName] = labelEncoder.transform(listName[columnName])
	except:
		listName[columnName] = modeDict[columnName] # encode the unknown value in train data using the mode
	return listName

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

	# preprocessing the input
	input_data = DataFrameImputer().fit_transform(input_data)

	# load the mode data
	pickle_in = open("static/model/mode_dict.pickle","rb")
	mode_dict = pickle.load(pickle_in)

	# label encode every column with string value
	input_data = labelEncode(input_data, 'workclass', mode_dict)
	input_data = labelEncode(input_data, 'education', mode_dict)
	input_data = labelEncode(input_data, 'marital-status', mode_dict)
	input_data = labelEncode(input_data, 'occupation', mode_dict)
	input_data = labelEncode(input_data, 'relationship', mode_dict)
	input_data = labelEncode(input_data, 'race', mode_dict)
	input_data = labelEncode(input_data, 'sex', mode_dict)
	input_data = labelEncode(input_data, 'native-country', mode_dict)

	# later should handle if the data is not in the label encoder

	# one hot encoding
	atributtest = input_data[["workclass", "education", "education-num", "marital-status", "occupation", "relationship", "sex", "capital-gain", "capital-loss"]].as_matrix()
	atributtemptest = pd.DataFrame(atributtest)
	atributtemptest.columns = ["workclass", "education", "education-num", "marital-status", "occupation", "relationship", "sex", "capital-gain", "capital-loss"]

	atributtemptest = oneHotEncode(atributtemptest, 'relationship', 0, 5)
	atributtemptest = oneHotEncode(atributtemptest, 'workclass', 1, 8)
	atributtemptest = oneHotEncode(atributtemptest, 'education', 0, 15)
	atributtemptest = oneHotEncode(atributtemptest, 'marital-status', 0, 6)
	atributtemptest = oneHotEncode(atributtemptest, 'occupation', 1, 14)

	inputPreprocessed = atributtemptest.as_matrix()

	prediction = model.predict(inputPreprocessed)
	if prediction[0] == 0:
		return '<=50K'
	else:
		return '>50K'