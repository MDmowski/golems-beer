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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Ewaluacja
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score

df = pd.read_csv('beers.csv')
df = df.drop(['UserId'], axis=1)
df = df.drop(['Name'], axis=1)
df = df.drop(['PitchRate'], axis=1)

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


# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 30, 50, 100],
    'max_features': [2, 3, 5],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

clf = RandomForestClassifier()
gridSearch = GridSearchCV(estimator=clf, param_grid=param_grid,
                          cv=3, n_jobs=-1, verbose=2)
gridSearch.fit(xTrain, yTrain)
print(gridSearch.best_params_)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(predictions, test_features)

    return accuracy


base_model = RandomForestClassifier()
base_model.fit(xTrain, yTrain)
base_predictions = base_model.predict(xTest)
base_accuracy = accuracy_score(base_predictions, yTest)
print(base_accuracy)

best_grid = gridSearch.best_estimator_
print(best_grid)
grid_predictions = best_grid.predict(xTest)
grid_accuracy = accuracy_score(grid_predictions, yTest)
print(grid_accuracy)

print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy)))
      