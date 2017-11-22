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
        '''Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        '''
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

def handleEmpty(listName, columnName):
	try:
		listName[columnName] = int(listName[columnName])
	except:
		listName[columnName] = -1
	return listName

def predictResult(model, age, workclass, fnlwgt, education, educationNumber,
	maritalStatus, occupation, relationship, race, sex,
	capitalGain, capitalLoss, hoursPerWeek, nativeCountry):
	input = [(age, workclass, fnlwgt, education, educationNumber,
		maritalStatus, occupation, relationship, race, sex,
		capitalGain, capitalLoss, hoursPerWeek, nativeCountry)]
	labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
		'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
		'hours-per-week', 'native-country']

	inputData = pd.DataFrame.from_records(input, columns=labels)

	# handle the empty value in integer valued columns
	inputData = handleEmpty(inputData, 'age')
	inputData = handleEmpty(inputData, 'fnlwgt')
	inputData = handleEmpty(inputData, 'education-num')
	inputData = handleEmpty(inputData, 'capital-gain')
	inputData = handleEmpty(inputData, 'capital-loss')
	inputData = handleEmpty(inputData, 'hours-per-week')

	# preprocessing the input
	inputData = DataFrameImputer().fit_transform(inputData)

	# load the mode data
	pickleIn = open('static/model/mode_dict.pickle','rb')
	modeDict = pickle.load(pickleIn)

	# label encode every column with string value
	inputData = labelEncode(inputData, 'workclass', modeDict)
	inputData = labelEncode(inputData, 'education', modeDict)
	inputData = labelEncode(inputData, 'marital-status', modeDict)
	inputData = labelEncode(inputData, 'occupation', modeDict)
	inputData = labelEncode(inputData, 'relationship', modeDict)
	inputData = labelEncode(inputData, 'race', modeDict)
	inputData = labelEncode(inputData, 'sex', modeDict)
	inputData = labelEncode(inputData, 'native-country', modeDict)

	# later should handle if the data is not in the label encoder

	# one hot encoding
	temporaryInputData = inputData[['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'sex', 'capital-gain', 'capital-loss']].as_matrix()
	temporaryInputDataFrame = pd.DataFrame(temporaryInputData)
	temporaryInputDataFrame.columns = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'sex', 'capital-gain', 'capital-loss']

	# encode using one hot encoding depends on the min range and max range in the columns' possible value
	temporaryInputDataFrame = oneHotEncode(temporaryInputDataFrame, 'relationship', 0, 5)
	temporaryInputDataFrame = oneHotEncode(temporaryInputDataFrame, 'workclass', 1, 8)
	temporaryInputDataFrame = oneHotEncode(temporaryInputDataFrame, 'education', 0, 15)
	temporaryInputDataFrame = oneHotEncode(temporaryInputDataFrame, 'marital-status', 0, 6)
	temporaryInputDataFrame = oneHotEncode(temporaryInputDataFrame, 'occupation', 1, 14)

	inputPreprocessed = temporaryInputDataFrame.as_matrix()

	prediction = model.predict(inputPreprocessed)
	return prediction[0]