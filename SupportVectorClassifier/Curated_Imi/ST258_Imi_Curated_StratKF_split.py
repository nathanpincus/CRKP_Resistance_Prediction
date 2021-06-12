# Splitting dataset into folds for manual cross-validaiton

# Load in modules needed for data import/processing
#For data inport/proccessing
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

# ### Import curated feature set and labels

# Import in curated feature set
Curated = pd.read_csv("/path/to/ST258_CuratedFeatures_Filtered.csv", index_col="genome")
Curated = Curated.astype("float64")
print("Accessory genome dataset (head):", flush=True)
print(Curated.head(), flush=True)
print("", flush=True)

# Import in labels.
# Genome name as index column
labels = pd.read_csv("/path/to/ST258_labels.csv", index_col="genome")
print("Labels:", flush=True)
print(labels.head(), flush=True)
print("", flush=True)

# Set seed for repeatable results (based on previous call of np.random.randint(low=1, high=1e9))
s = 731816214
print("Seed used in this analysis is " + str(s), flush=True)

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

# Define X and y
Curated["rank"] = labels["imipenem_gt8"]

# Define X as all features and y as all labels
X = Curated.drop("rank", axis=1)
y = Curated["rank"]
print("Features:", flush=True)
print(X.head(), flush=True)
print("Labels:", flush=True)
print(y, flush=True)


# Split into folds using stratkf and save these folds to files
split = 0
for train_index, test_index in stratkf.split(X, y):
    print("Split {0}".format(str(split)), flush=True)
    Xtrain = X.iloc[train_index,]
    ytrain = y.iloc[train_index,]
    Xtest = X.iloc[test_index,]
    print("Test Features:", flush=True)
    print(Xtest.head(), flush=True)
    ytest = y.iloc[test_index,]
    print("Test Labels", flush=True)
    print(ytest.head(), flush=True)
    with open('ST258_Imi_Curated_SVC_f1Tuned_Xtrain_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(Xtrain, f)
    with open('ST258_Imi_Curated_SVC_f1Tuned_ytrain_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(ytrain, f)
    with open('ST258_Imi_Curated_SVC_f1Tuned_Xtest_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(Xtest, f)
    with open('ST258_Imi_Curated_SVC_f1Tuned_ytest_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(ytest, f)
    split = split + 1
