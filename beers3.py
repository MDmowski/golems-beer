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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
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

Y = df['Style'].values
lb = LabelEncoder()
Y = lb.fit_transform(Y)
dataFile = df.drop(['Style'], axis=1)
X = dataFile.values
xTrain, xTest, yTrain, yTest = train_test_split(
    X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(X)

xTrain = pd.DataFrame(imp.transform(xTrain))
xTest = pd.DataFrame(imp.transform(xTest))

randTree = RandomForestClassifier(
    n_estimators=100, max_depth=7, random_state=42, warm_start=True)
randTree.fit(xTrain, yTrain)
yPred = randTree.predict(xTest)
accuracy = accuracy_score(yPred, yTest)
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

yTestPred = randTree.predict(testFileModified)
output = pd.DataFrame({'Id': idTestFile, 'Style': yTestPred})
output.to_csv('output3.csv', index=False)
