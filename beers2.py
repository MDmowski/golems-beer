# Analiza i modyfikacja danych
from sklearn.experimental import enable_iterative_imputer
from sklearn import tree
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
# Wizualizacja
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Ewaluacja
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score

df = pd.read_csv('beers.csv')
df = df.drop(['UserId'], axis=1)
df = df.drop(['Name'], axis=1)
df = df.drop(['PitchRate'], axis=1)
styles = df['Style'].unique()

trainMask = np.random.rand(len(df)) < 0.7
dfTrain = df[trainMask]
dfTest = df[~trainMask]

accuracies =[]

#print(dfTrainImp.info())
folds = KFold(n_splits=5).split(dfTrain)
for TrainIdx, TestIdx in folds:
        dfTrainFold = df.iloc[TrainIdx]
        dfTestFold = df.iloc[TestIdx]
        dfTrainFold = dfTrainFold.fillna(dfTrainFold.mean())
        dfTestFold = dfTestFold.fillna(dfTrainFold.mean())

        xTrainFold = dfTrainFold.loc[:, dfTrainFold.columns != 'Style']
        yTrainFold = dfTrainFold.loc[:, dfTrainFold.columns == 'Style']
        xTestFold = dfTestFold.loc[:, dfTestFold.columns != 'Style']
        yTestFold = dfTestFold.loc[:, dfTestFold.columns == 'Style']

        clf = tree.DecisionTreeClassifier(max_depth=7)
        clf = clf.fit(xTrainFold, yTrainFold)
        yPredFold = clf.predict(xTestFold)
        accuracy = accuracy_score(yPredFold, yTestFold)

testFile = pd.read_csv('beers_test_nostyle.csv')
idTestFile = testFile['Id'].values
testFile = testFile.drop(['Id'], axis=1)
testFile = testFile.drop(['UserId'], axis=1)
testFile = testFile.drop(['PitchRate'], axis=1)

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(testFile)
testFileModified = pd.DataFrame(imp.transform(testFile))
testFileModified.columns = testFile.columns


# testFileModified = testFile.fillna(testFile.mean())

testFileValues = testFileModified.values

yTestPred = clf.predict(testFileValues)
output = pd.DataFrame({'Id': idTestFile, 'Style': yTestPred})
output.to_csv('output2.csv', index=False)
