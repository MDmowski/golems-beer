# Analiza i modyfikacja danych
from sklearn.experimental import enable_iterative_imputer
from sklearn import tree
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
import keras
# Wizualizacja
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from keras import backend as K
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GaussianDropout, AlphaDropout

# Ewaluacja
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score

# Wczytanie danych
df = pd.read_csv('beers.csv')
# df = df.drop(df.columns[0], axis=1)
df = df.drop(['UserId'], axis=1)
df = df.drop(['Name'], axis=1)
df = df.drop(['PitchRate'], axis=1)

# Kodowanie Y
categoricals = list(df.select_dtypes(include=['O']).columns)
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[categoricals])
train_ohe = pd.DataFrame(encoded, columns=np.hstack(encoder.categories_))
df = pd.concat((df, train_ohe), axis=1).drop(categoricals, axis=1)
# print(df.head())

# Podział na X i Y
Y = df.iloc[:, -7:]
df = df.drop(df.columns[-7:], axis=1)
X = df
styles = list(Y.columns.values.tolist())
print(styles)
# Podział na testowy i treningowy
xTrain, xTest, yTrain, yTest = train_test_split(
    X, Y, test_size=0.2, random_state=1337)
# print(yTrain)

# Uzupełnienie brakujących wartości
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(X)
xTrain = pd.DataFrame(imp.transform(xTrain))
xTest = pd.DataFrame(imp.transform(xTest))
# Trenowanie modelu
model = Sequential()
model.add(Dense(28, activation='relu',
                kernel_initializer='random_normal', input_dim=8))
model.add(Dense(17, kernel_initializer='random_normal', activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=100, batch_size=128)
score = model.evaluate(xTest, yTest, batch_size=128)
print(score)

# Wczytanie danych z pliku testowego
testFile = pd.read_csv('beers_test_nostyle.csv')
idTestFile = testFile['Id'].values
testFile = testFile.drop(['Id'], axis=1)
testFile = testFile.drop(['UserId'], axis=1)
testFile = testFile.drop(['PitchRate'], axis=1)

# Uzupełnienie danych w pliku testowym
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(testFile)
testFileModified = pd.DataFrame(imp.transform(testFile))
testFileModified.columns = testFile.columns
testFileValues = testFileModified.values

# Przewidywanie wartości
yTestPred = model.predict(testFileModified)
yTestPred = [np.argmax(t) for t in yTestPred]
outcomes = []
for pred in yTestPred:
    outcomes.append(styles[pred])

output = pd.DataFrame({'Id': idTestFile, 'Style': outcomes})
output.to_csv('output4.csv', index=False)
