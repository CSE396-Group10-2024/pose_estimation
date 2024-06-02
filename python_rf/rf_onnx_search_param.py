from sklearn import datasets
import pandas as pd
import numpy as np
import socket
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import json
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
# Load and preprocess data
path = '/home/buysal/396/svm/onnx_csv'
dfs=[]
all_files = glob.glob(path + "/*.csv")
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath)
            # Drop the 'index' column if it exists
            if 'index' in df.columns:
                df = df.drop(columns=['index'])
            # Drop columns that include 'angle' in their name
         

            dfs.append(df)
features = pd.concat(dfs, ignore_index=True)
print(features.columns)
class_counts = features['target_class'].value_counts()
print(class_counts)
#   {"left_arm_body",9,11,0},
#     {"right_arm_body",10,12,0},
    
#     {"right_elbow_right_head",8,4,0},
#     {"left_elbow_left_head",7,3,0}

def find_columns_to_drop(df):
    columns_to_drop = [col for col in df.columns if (
        ('z' in col and len(col) in [2, 3]) or 
        ('x' in col and len(col) in [2, 3])or
        ('y' in col and len(col) in [2, 3]) or
        ('visibility' in col) or 
        ('presence' in col) or 
        ('localized' in col) or
        ('_loc' in col) or
        ('Unnamed'in col)

    )]
    return columns_to_drop

# Get the columns that meet the criteria
columns_to_drop = find_columns_to_drop(features)

# Drop the identified columns
features = features.drop(columns=columns_to_drop)

print(features.columns)
duplicate_rows = features.duplicated()
num_duplicates = duplicate_rows.sum()
print(f'Number of duplicate rows: {num_duplicates}')
duplicates = features[duplicate_rows]
print('Duplicate rows:')
print(duplicates)
features = features.drop_duplicates()

duplicate_rows = features.duplicated()
num_duplicates = duplicate_rows.sum()
print(f'Number of duplicate rows: {num_duplicates}')

X = features.drop(['target_class'], axis=1)
y = features['target_class']
cols = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
cols = X_train.columns

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)


param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'bootstrap': [True, False]
}

# Initialize the model
clf = RandomForestClassifier()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, 
                                   n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", random_search.best_params_)
# Replace SVC with RandomForestClassifier
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))

