#!/usr/bin/python

import pickle

from sklearn.pipeline import Pipeline
from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from tester import main
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred',
                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi',
                 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key in data_dict:
    total_payments = data_dict[key]['total_payments']
    total_stock_value = data_dict[key]['total_stock_value'] if data_dict[key]['total_stock_value'] != 'NaN' else 0.0
    stocks_vs_payments = float(total_stock_value) / total_payments if total_payments != 'NaN' else 0.0
    data_dict[key]['stocks_vs_payments'] = stocks_vs_payments

for key in data_dict:
    total_payments = data_dict[key]['total_payments']
    total_salary = data_dict[key]['salary'] if data_dict[key]['salary'] != 'NaN' else 0.0
    salary_vs_payments = float(total_salary) / total_payments if total_payments != 'NaN' else 0.0
    data_dict[key]['salary_vs_payments'] = salary_vs_payments

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features = np.array(features)
labels = np.array(labels)
fss = SelectKBest(score_func=f_classif)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
clf = DecisionTreeClassifier(random_state=42)
pipe = Pipeline(steps=[('fss', fss), ('clf', clf)])

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
cv = StratifiedShuffleSplit(labels, n_iter=50, random_state=42)
param_grid = [
  {'clf__min_samples_split': [2, 5, 10, 15, 20, 50], 'clf__criterion': ['gini', 'entropy'],
   'clf__max_features': ['sqrt', 'log2', None], 'clf__class_weight': [None, 'balanced'],
   'clf__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 20, 50],
   'fss__k': range(1, len(features_list))}
 ]
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', n_jobs=1)
grid.fit(features, labels)
clf = grid.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
main()
