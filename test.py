from azureml.train.automl import AutoMLConfig
import logging
import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.model import Model
from azureml.core.webservice import Webservice
import azureml.train.automl
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import joblib

ws = Workspace.get(
    name="golem", subscription_id='cde71551-cef6-45b7-aecc-acc8b738c6a0', resource_group='golem')

dataFile = pd.read_csv('beers.csv')
dataFile = dataFile.drop(['UserId'], axis=1)
dataFile = dataFile.drop(['Name'], axis=1)
dataFile = dataFile.drop(['PitchRate'], axis=1)

y_df = dataFile.pop("Style")
x_df = dataFile
print(x_df)
x_train, x_test, y_train, y_test = train_test_split(
    x_df, y_df, test_size=0.2, random_state=223)


automl_settings = {
    "enable_early_stopping": True,
    "primary_metric": 'accuracy',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations": 5
}


automl_config = AutoMLConfig(task='classification',
                             debug_log='automated_ml_errors.log',
                             X=x_train.values,
                             y=y_train.values.flatten(),
                             **automl_settings)
experiment = Experiment(ws, "beer-classification")
local_run = experiment.submit(automl_config, show_output=True)

best_run, fitted_model = local_run.get_output()
print(best_run)
print(fitted_model)

testFile = pd.read_csv('beers_test_nostyle.csv')
idTestFile = testFile['Id'].values
testFile = testFile.drop(['Id'], axis=1)
testFile = testFile.drop(['UserId'], axis=1)
testFile = testFile.drop(['PitchRate'], axis=1)
y_predict = fitted_model.predict(testFile.values)
output = pd.DataFrame({'Id': idTestFile, 'Style': y_predict})
output.to_csv('azure.csv', index=False)
