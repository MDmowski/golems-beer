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
from scipy import stats

from sklearn.preprocessing import OneHotEncoder

dataFile = pd.read_csv('beers.csv')
df = dataFile.drop(['Name'], axis=1)
df = df.drop(['UserId'], axis=1)
df = df.drop(['PitchRate'], axis=1)
# df = df.drop(['Style'], axis=1)

stouts = df.loc[df['Style'] == 'Stout']
ales = df.loc[df['Style'] == 'Ale']
lagers = df.loc[df['Style'] == 'Lager']
ipas = df.loc[df['Style'] == 'IPA']
witbiers = df.loc[df['Style'] == 'Witbier']
porters = df.loc[df['Style'] == 'Porter']
saisons = df.loc[df['Style'] == 'Saison']

imp = IterativeImputer(max_iter=10, random_state=0)

stouts = stouts.drop(['Style'], axis=1)
imp.fit(stouts)
filled = pd.DataFrame(imp.transform(stouts))
filled.columns = stouts.columns
z = np.abs(stats.zscore(filled))
filled = filled[(z < 3).all(axis=1)]
filled['Style'] = 'Stout'
stouts = filled


ales = ales.drop(['Style'], axis=1)
imp.fit(ales)
filled = pd.DataFrame(imp.transform(ales))
filled.columns = ales.columns
z = np.abs(stats.zscore(filled))
filled = filled[(z < 3).all(axis=1)]
filled['Style'] = 'Ale'
ales = filled

lagers = lagers.drop(['Style'], axis=1)
imp.fit(lagers)
filled = pd.DataFrame(imp.transform(lagers))
filled.columns = lagers.columns
z = np.abs(stats.zscore(filled))
filled = filled[(z < 3).all(axis=1)]
filled['Style'] = 'Lager'
lagers = filled

ipas = ipas.drop(['Style'], axis=1)
imp.fit(ipas)
filled = pd.DataFrame(imp.transform(ipas))
filled.columns = ipas.columns
z = np.abs(stats.zscore(filled))
filled = filled[(z < 3).all(axis=1)]
filled['Style'] = 'IPA'
ipas = filled

witbiers = witbiers.drop(['Style'], axis=1)
imp.fit(witbiers)
filled = pd.DataFrame(imp.transform(witbiers))
filled.columns = witbiers.columns
z = np.abs(stats.zscore(filled))
filled = filled[(z < 3).all(axis=1)]
filled['Style'] = 'Witbier'
witbiers = filled

porters = porters.drop(['Style'], axis=1)
imp.fit(porters)
filled = pd.DataFrame(imp.transform(porters))
filled.columns = porters.columns
z = np.abs(stats.zscore(filled))
filled = filled[(z < 3).all(axis=1)]
filled['Style'] = 'Porter'
porters = filled

saisons = saisons.drop(['Style'], axis=1)
imp.fit(saisons)
filled = pd.DataFrame(imp.transform(saisons))
filled.columns = saisons.columns
z = np.abs(stats.zscore(filled))
filled = filled[(z < 3).all(axis=1)]
filled['Style'] = 'Saison'
saisons = filled

filled = pd.concat([stouts, ales, lagers, ipas, witbiers,
                    porters, saisons], ignore_index=1)
filled = filled.sample(frac=1).reset_index(drop=True)
print(filled)
filled.to_csv('preprocessed.csv')
