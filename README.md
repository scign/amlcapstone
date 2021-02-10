# Winemaking used to be an art. AI has turned it into a science.

You can't directly measure the "quality" of a good wine since quality is a subjective measure. However this quality is based on a number of properties of the wine, many of them physical and chemical in nature. Therefore you can measure those properties and model how those properties correlate to the subjective "quality".

We looked at how AutoML and Hyperdrive assist with exploring different model types and hyperparameters, to find a model that can predict this subjective wine "quality" from the measurable physicochemical properties such as pH and sulfate content.

Both methods produced models with a small Normalized Root Mean Squared Error (NRMSE), interpreted as just 12-13% of the target variable range which is apparently pretty good. The AutoML model outperformed the Hyperdrive model slightly. We deployed the AutoML model and validated that the model was deployed by sending an HTTP request to the model endpoint and receiving a valid response.

Suggestions for improvements were identified and discussed below.

## Dataset

### Overview
We will be using the Wine Quality dataset made accessible [here](https://archive.ics.uci.edu/ml/datasets/wine+quality).

The dataset contains the physicochemical properties of 1599 red wine samples. The data includes, in column order: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`. The data also includes, as the final column, the target variable `quality` (score between 0 and 10) which was human-derived.

### Task
We will be using all features except `quality`, to try and predict wine quality. The original research treated this as a classification task, however since all the features are numerical and ordered, we hypothesize whether a regression model may be able to predict quality given the input features.

### Access
The data is made accessible through a public link to the UCI data archive.
* For AutoML we create a dataset using the dataset URL and register the dataset in the workspace. We then pull the dataset into a pandas dataframe and split the dataset into train and test dataframes. The train dataset is exported to a CSV file, uploaded to the default datastore, and a DataSet object is constructed that points to the file in the datastore. This is done so that we end up with a pointer that we can pass to the AutoML engine, which is good practice.
* For Hyperdrive we access the data directly from the `train.py` script, ultimately getting a dataframe with the contents, again splitting that into train and test segments, and using the train dataset to build the model.

## Automated ML
Given that this is a regression task, we chose the normalized root mean squared error (NRMSE) as the primary metric to optimize. This calculates the mean difference between sample targets and predicted values, as a proportion of the target range. We are looking to minimize this value, i.e. look for the model with the smallest overall difference between predicted value and target value.

The following configuration settings were used:
Setting | Value | Comments
- | - | -
experiment_timeout_minutes | 15 | Modelling this small dataset is very quick and we should be able to train a significant number of models in this time.
max_concurrent_iterations | 5 | Ensuring that we make good use of the compute instance nodes
n_cross_validations | 5 | Cross validation ensures that we achieve sufficient coverage of the dataset
primary_metric | normalized_root_mean_squared_error | This is how we will judge how well the model fits the data


### Results
The AutoML Run is shown below.

![AutoML run details](assets/automl-run.png)

The last two models trained are the StackEnsemble and VotingEnsemble. As usual these are the best two models in the group.

These are the best model identified by the AutoML run.

![AutoML best models](assets/automl-best-models.png)

The best model:

![AutoML best model](assets/automl-best-model.png)

Components and parameters of the best model:

![Best model (1 of 2)](assets/best_model1.png)
![Best model (2 of 2)](assets/best_model2.png)

The VotingEnsemble was the better of the two, and yielded a NRMSE of 0.1207. The ensemble model consisted of the following models:
* LightGBMRegressor (run 0, weight 7/15) - best model apart from the ensembles, given the highest weighting
* GradientBoostingRegressor (run 34, weight 1/15)
* RandomForestRegressor (run 29, weight 1/15)
* RandomForestRegressor (run 33, weight 1/15)
* GradientBoostingRegressor (run 32, weight 2/15)
* ExtraTreesRegressor (run 26, weight 1/15)
* DecisionTreeRegressor (run 3, weight 1/15)
* DecisionTreeRegressor (run 22, weight 1/15)

## Hyperparameter Tuning

## Hyperdrive Configuration
For the Hyperdrive run, we explored the parameter space of an ElasticNet model. ElasticNet is a linear model that combines L1 and L2 regularization. The parameters `alpha` and `l1_ratio` control the regularization penalties. Both are continuous parameters. 

The `l1_ratio` is a measure of how much the model uses L1 vs L2 regularization. This parameter varies between 0 (100% L1) and 1 (100% L2). Based on some prior cursory review of the variables, not documented here, some of the input features appear to vary relatively linearly with the target variable, therefore we should expect a better model towards the L1 end of the spectrum. We will set this parameter to vary across the full 0-1 range to validate this hypothesis.

`alpha` is a multiplier of the penalty terms which reduce overfitting. Given the data, overregularization may underfit the model too much so we expect this to be low. Given that it is a multiplier we set the range to be `loguniform` across the 1x10^-4 to 1x10^-2 space.

The Bandit early stopping policy with a 20% slack ratio terminates runs if the primary metric does not match the best run so far within 20% (e.g. if the best run so far had a NRMSE of 0.5, any run with a NRMSE above 0.6 will be terminated). This is suitable for a regression task.

### Results
![Hyperdrive run details](assets/hyperdrive-run.png)

The scatter plot on the right shows the 2d parameter space, with the hue as the NMRSE. Darker is better. The plot gets darker towards the origin. As expected, the lowest `alpha` and `l1_ratio` values produced the model with the lowest error.

![Hyperdrive best models](assets/hyperdrive-best-models.png)

The model with the lowest NMRSE was trained with `alpha`=0.0344 and `l1_ratio`=0.0714.

![Hyperdrive best model](assets/hyperdrive-best-model.png)

### Best model between Hyperdrive and AutoML
The AutoML run produced a model with a lower NMRSE than the Hyperdrive run (0.12073 vs 0.13032). We will take this as the best model.

## Model Deployment
The deployed model endpoint is shown here:
![Deployed model](assets/endpoint.png)

The following code can be used to query the model.
```
import json
import requests
import pandas as pd

scoring_uri = ''  # insert the URL to the deployed endpoint

data_file_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(data_file_source, delimiter=';').dropna()
# select a few random rows from the test set to score
random_data = df.sample(5, random_state=42).values
x_test = random_data[:,:-1].tolist()
y_test = random_data[:,-1].tolist()

input_data = "{\"data\": " + str(x_test) + "}"
headers = {'Content-Type':'application/json'}

resp = requests.post(scoring_uri, input_data, headers=headers)

print("POST to url", scoring_uri)
print("input data:", input_data)
print("label:", y_test)
print("prediction:", resp.text)
```
![Endpoint demo](assets/deployed_endpoint_response.png)

To ensure consistency of the environment if this model needed to be deployed again, two files, `azureml_environment.json` and `conda_dependencies.yml` are made available through the Environment object constructed for deployment. The files representing the environment used here are at the following links:
* [azureml_environment.json](assets/azureml_environment.json)
* [conda_dependencies.yml](assets/conda_dependencies.yml)

## Screen Recording
A screencast showing the following is published on [YouTube](https://youtu.be/1U_jEPqlKUI). I forgot to zoom in on the notebooks so as long as you review the screencast in fullscreen mode, and select 1080p quality, the content will be clear. The video is uploaded in 1080p.
- The working model
- Demo of the deployed model
- Demo of a sample request sent to the endpoint and its response

## Improvements
### Imbalance
<div style="bg-white">
  <img src="assets/imbalance.png" />
</div>

Over 80% of the samples in the dataset are given a quality rating of 5 or 6 and since the range is from 3 to 8, the bulk of the samples is centrally situated. The large central data mass drastically reduces the influence of peripheral samples, meaning that a large range of models could fit the data with a high perceived accuracy. This is apparent when looking at the residual plot - the model has failed to weight samples with quality at the ends of the range sufficiently, to identify the relationship correctly.

<div style="bg-white">
  <img src="assets/residual.png" />
</div>

Three things that could mitigate this are:
1. Reviewing the model test samples to check whether the model accurately fit the peripheral samples;
1. Reducing the number of samples with a 5/6 rating to reduce the influence of the central mass and allow the model to fit the peripheral data better;
1. Undersampling from quality ratings 5 and 6 or using methods such as SMOTE to oversample the other quality groups; or
1. Try treating it as a classification task and using the same undersampling/oversampling methods.

### Web front end
Accessing the deployed model currently requires constructing an HTTP POST request with the input features and parsing the response JSON. A graphical user interface would make this process more user friendly. A form could be constructed with fields allowing the user to enter the measured properties of the wine and obtain the quality rating when the form is submitted.

### IoT deployment
All these metrics can be measured by sensors. This means that the model can be deployed on an IoT device and connected to sensors to measure these properties in an industrial setting to monitor wine quality in real time. This would provide the winemaker with:
1. A way to monitor when quality is increasing or decreasing
1. Feedback on the production and fermentation/maturation methods to identify possible improvements
1. The optimal time to sell or serve their wine
1. Identification of defective batches for an improved product consistency and customer experience
