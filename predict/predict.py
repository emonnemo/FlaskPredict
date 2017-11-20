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

	# categorize the string attributes to integer
	input_data["workclass"] = input_data["workclass"].astype('category')
	input_data["education"] = input_data["education"].astype('category')
	input_data["marital-status"] = input_data["education"].astype('category')
	input_data["occupation"] = input_data["occupation"].astype('category')
	input_data["relationship"] = input_data["relationship"].astype('category')
	input_data["race"] = input_data["race"].astype('category')
	input_data["sex"] = input_data["sex"].astype('category')
	input_data["native-country"] = input_data["native-country"].astype('category')

	input_data["workclass"] = input_data["workclass"].cat.codes
	input_data["education"] = input_data["education"].cat.codes
	input_data["marital-status"] = input_data["marital-status"].cat.codes
	input_data["occupation"] = input_data["occupation"].cat.codes
	input_data["relationship"] = input_data["relationship"].cat.codes
	input_data["race"] = input_data["race"].cat.codes
	input_data["sex"] = input_data["sex"].cat.codes
	input_data["native-country"] = input_data["native-country"].cat.codes

	# return str(input_list)
	# return str(input_data)
	# return str(input_data.loc[:])
	prediction = model.predict(input_data)
	if prediction[0] == 0:
		return '<=50K'
	else:
		return '>50K'
	# return str(model.predict(input_data.loc[:]))
	# return str(model.predict([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])[0])