# Analiza i modyfikacja danych
from xgboost import XGBClassifier
from numpy import loadtxt
from sklearn.experimental import enable_iterative_imputer
from sklearn import tree
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


# Wczytanie danych
df = pd.read_csv('preprocessed.csv')
df = df.drop(df.columns[0], axis=1)
# df = df.drop(['UserId'], axis=1)
# df = df.drop(['Name'], axis=1)
# df = df.drop(['PitchRate'], axis=1)


trainMask = np.random.rand(len(df)) < 0.7
dfTrain = df[trainMask]
dfTest = df[~trainMask]

a = XGBClassifier()
parameters = {
    "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
}


# fit model no training data
folds = KFold(n_splits=7).split(dfTrain)
for TrainIdx, TestIdx in folds:
    dfTrainFold = df.iloc[TrainIdx]
    dfTestFold = df.iloc[TestIdx]
    dfTrainFold = dfTrainFold.fillna(dfTrainFold.mean())
    dfTestFold = dfTestFold.fillna(dfTrainFold.mean())

    xTrainFold = dfTrainFold.loc[:, dfTrainFold.columns != 'Style']
    yTrainFold = dfTrainFold.loc[:, dfTrainFold.columns == 'Style']
    xTestFold = dfTestFold.loc[:, dfTestFold.columns != 'Style']
    yTestFold = dfTestFold.loc[:, dfTestFold.columns == 'Style']
    clf = XGBClassifier(objective='multi:softmax')
    clf.fit(xTrainFold, yTrainFold)
    yPredFold = clf.predict(xTestFold)
    accuracy = accuracy_score(yPredFold, yTestFold)
    print(accuracy)

testFile = pd.read_csv('beers_test_nostyle.csv')
idTestFile = testFile['Id'].values
testFile = testFile.drop(['Id'], axis=1)
testFile = testFile.drop(['UserId'], axis=1)
testFile = testFile.drop(['PitchRate'], axis=1)

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(testFile)
testFileModified = pd.DataFrame(imp.transform(testFile))
testFileModified.columns = testFile.columns

testFileValues = testFileModified.values

yTestPred = clf.predict(testFileModified)
output = pd.DataFrame({'Id': idTestFile, 'Style': yTestPred})
output.to_csv('output5.csv', index=False)
