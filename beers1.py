# Analiza i modyfikacja danych
from sklearn.utils import resample
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
from sklearn.linear_model import LinearRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Ewaluacja
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score


random.seed(71830)

# wczytujemy dane
dataFile = pd.read_csv('beers.csv')
# dataFile = dataFile.drop(dataFile.columns[0], axis=1)
dataFile = dataFile.drop(['UserId'], axis=1)
dataFile = dataFile.drop(['Name'], axis=1)
dataFile = dataFile.drop(['PitchRate'], axis=1)
styles = dataFile['Style'].unique()
print(dataFile.info())

# #patrzymy jak skorelowane są dane numeryczne
# sns.heatmap(dataFile.corr(), annot=True)
# plt.tight_layout()
# plt.show()

Y = dataFile['Style'].values
dataFile = dataFile.drop(['Style'], axis=1)
print(Y)
# categoricals = list(dataFile.select_dtypes(include=['O']).columns)
# encoder = OneHotEncoder(sparse=False)
# encoded = encoder.fit_transform(dataFile[categoricals])
# train_ohe = pd.DataFrame(encoded, columns=np.hstack(encoder.categories_))
# dataFile = pd.concat((dataFile, train_ohe), axis=1).drop(categoricals, axis=1)

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# imp = IterativeImputer(max_iter=10, random_state=0)
# imp.fit(dataFile)

# dataFileFilled = pd.DataFrame(imp.transform(dataFile))
# dataFileFilled.columns = dataFile.columns
# X = dataFileFilled.values

X = dataFile.values
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(dataFile)

X_train_fill = pd.DataFrame(imp.transform(X_train))
X_train_fill.columns = dataFile.columns

X_test_fill = pd.DataFrame(imp.transform(X_test))
X_test_fill.columns = dataFile.columns

clf = GradientBoostingClassifier(max_depth=10)
clf = clf.fit(X_train_fill, y_train)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(300 , 100))
# tree.plot_tree(clf, filled=True, feature_names=dataFile.columns, class_names=(styles))
# plt.savefig('tree.png')

y_pred = clf.predict(X_test_fill)
accuracy = sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.3f}")

testFile = pd.read_csv('beers_test_nostyle.csv')
idTestFile = testFile['Id'].values
print(idTestFile)
testFile = testFile.drop(['Id'], axis=1)
testFile = testFile.drop(['UserId'], axis=1)
testFile = testFile.drop(['PitchRate'], axis=1)

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(testFile)

testFileModified = pd.DataFrame(imp.transform(testFile))
testFileModified.columns = testFile.columns

testFileValues = testFileModified.values


yTestPred = clf.predict(testFileValues)
print(yTestPred)
output = pd.DataFrame({'Id': idTestFile, 'Style': yTestPred})
output.to_csv('output1.csv', index=False)
