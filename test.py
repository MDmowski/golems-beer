import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from azureml.widgets import RunDetails
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from azureml.train.automl import AutoMLConfig
import logging
import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.model import Model
from azureml.core.webservice import Webservice
from azureml.core.dataset import Dataset
import azureml.train.automl
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import logging
from matplotlib import pyplot as plt
import joblib

ws = Workspace.get(
    name="golem", subscription_id='cde71551-cef6-45b7-aecc-acc8b738c6a0', resource_group='golem')

cts = ws.compute_targets
amlcompute_cluster_name = "golem"

found = False
if amlcompute_cluster_name in cts and cts[amlcompute_cluster_name].type == 'cpu-cluster-1':
    found = True
    print('Found existing compute target.')
    compute_target = cts[amlcompute_cluster_name]

if not found:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_DS12_V2",  # for GPU, use "STANDARD_NC6"
                                                                vm_priority='lowpriority',
                                                                max_nodes=6)
compute_target = ComputeTarget.create(
    ws, amlcompute_cluster_name, provisioning_config)

print('Checking cluster status...')
# Can poll for a minimum number of nodes and for a specific timeout.
# If no min_node_count is provided, it will use the scale settings for the cluster.
compute_target.wait_for_completion(
    show_output=True, min_node_count=None, timeout_in_minutes=20)

data = "beers-mod.csv"
dataset = Dataset.Tabular.from_delimited_files(data)


training_data, validation_data = dataset.random_split(percentage=0.8, seed=223)


labelName = 'Style'

automl_settings = {
    "primary_metric": 'accuracy',
    "preprocess": True,
    "enable_early_stopping": True,
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations": 5,
    "enable_cache": True,
    "max_concurrent_iterations": 6
}


automl_config = AutoMLConfig(task='classification',
                             debug_log='automated_ml_errors.log',
                             compute_target=compute_target,
                             training_data=training_data,
                             label_column_name=labelName,
                             ** automl_settings)
experiment = Experiment(ws, "beer-classification")
# local_run = experiment.submit(automl_config, show_output=True)

remote_run = experiment.submit(automl_config, show_output=True)

RunDetails(remote_run).show()
remote_run.wait_for_completion(show_output=True)

best_run, fitted_model = remote_run.get_output()

X_test_df = validation_data.drop_columns(
    columns=[labelName]).to_pandas_dataframe()
y_test_df = validation_data.keep_columns(
    columns=[labelName], validate=True).to_pandas_dataframe()

y_pred = fitted_model.predict(X_test_df)


cf = confusion_matrix(y_test_df.values, y_pred)
plt.imshow(cf, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
class_labels = ['False', 'True']
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks([-0.5, 0, 1, 1.5], ['', 'False', 'True', ''])
# plotting text value inside cells
thresh = cf.max() / 2.
for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
    plt.text(j, i, format(cf[i, j], 'd'), horizontalalignment='center',
             color='white' if cf[i, j] > thresh else 'black')
plt.show()

testFile = pd.read_csv('beers_test_nostyle.csv')
idTestFile = testFile['Id'].values
testFile = testFile.drop(['Id'], axis=1)
testFile = testFile.drop(['UserId'], axis=1)
y_predict = fitted_model.predict(testFile.values)
output = pd.DataFrame({'Id': idTestFile, 'Style': y_predict})
output.to_csv('azure.csv', index=False)
