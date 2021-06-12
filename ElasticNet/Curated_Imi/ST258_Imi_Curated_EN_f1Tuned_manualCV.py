# ### Load in modules needed for data inport/proccessing, analysis, and plotting
import argparse

#For data inport/proccessing
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

# ### Load in command line arguments
parser = argparse.ArgumentParser(description="Manually determine cross-valdidation performance using pre-defined folds")
parser.add_argument("-s", "--split", required=True, help="current split")
args = parser.parse_args()

print("Current split is {0}".format(str(args.split)))

# ### Load in datasets
print("Training set:", flush=True)
with open("ST258_Imi_Curated_EN_f1Tuned_Xtrain_split{0}.pkl".format(str(args.split)), "rb") as f:
    Xtrain = pickle.load(f)
print(Xtrain.head(), flush=True)
with open("ST258_Imi_Curated_EN_f1Tuned_ytrain_split{0}.pkl".format(str(args.split)), "rb") as f:
    ytrain = pickle.load(f)
print(ytrain.head(), flush=True)

print("Test set:", flush=True)
with open("ST258_Imi_Curated_EN_f1Tuned_Xtest_split{0}.pkl".format(str(args.split)), "rb") as f:
    Xtest = pickle.load(f)
print(Xtrain.head(), flush=True)
with open("ST258_Imi_Curated_EN_f1Tuned_ytest_split{0}.pkl".format(str(args.split)), "rb") as f:
    ytest = pickle.load(f)
print(ytest.head(), flush=True)

# Set seed for repeatable results (based on previous call of np.random.randint(low=1, high=1e9))
s = 216826829
print("Seed used in this analysis is " + str(s), flush=True)

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

# Instantiate EN estimator
EN_model = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000,random_state=s)

# Create GSCV estimator
param_grid = {'C': [10 ** x for x in range(-4,5)],
              'l1_ratio': [0.1 * x for x in range(11)],
              'class_weight': ['balanced', None]}
GSCV_Curated = GridSearchCV(EN_model, param_grid, scoring="f1", cv=stratkf, iid=False, n_jobs=-1)

# Fit estimator to training data for this fold
print("Fit estimator to training dataset and show best parameters")
GSCV_Curated.fit(Xtrain,ytrain)
print(GSCV_Curated, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal score: " + str(GSCV_Curated.best_score_), flush=True)

#Print best parameters
print("Best parameters:", flush=True)
print(GSCV_Curated.best_params_, flush=True)

print("Complete CV results:", flush=True)
print(GSCV_Curated.cv_results_, flush=True)

# Pickle model for fold
with open('ST258_Imi_Curated_EN_f1Tuned_split{0}_GridSearch.pkl'.format(str(args.split)), 'wb') as f:
    pickle.dump(GSCV_Curated, f)

# Test against CV test set for this fold
print("Test performance against test set for this fold", flush=True)
# Predict values for the testing data
ypred = GSCV_Curated.predict(Xtest)
y_pred_prob = GSCV_Curated.predict_proba(Xtest)[:, 1]
# Determining perforance
Acc = metrics.accuracy_score(ytest, ypred)
Sen = metrics.recall_score(ytest, ypred)
Sp = metrics.recall_score(ytest, ypred, pos_label=0)
PPV = metrics.precision_score(ytest, ypred)
AUC = metrics.roc_auc_score(ytest, y_pred_prob)
F1 = metrics.f1_score(ytest, ypred)

# Save CV results for this fold to a file
Scores = pd.DataFrame(columns=["Accuracy", "Sensitivity", "Specificity", "PPV", "AUC", "F1"])
Scores.loc[0] = [Acc,Sen,Sp,PPV,AUC,F1]
print(Scores, flush=True)
Scores.to_csv("ST258_Imi_Curated_EN_f1Tuned_split{0}_CVResults.csv".format(str(args.split)), index=False)
