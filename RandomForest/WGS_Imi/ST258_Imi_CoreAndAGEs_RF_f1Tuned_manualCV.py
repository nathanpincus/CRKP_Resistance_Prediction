# ### Load in modules needed for data inport/proccessing, analysis, and plotting
import argparse

#For data inport/proccessing
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.ensemble import RandomForestClassifier
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
with open("ST258_Imi_CoreAndAGEs_RF_f1Tuned_Xtrain_split{0}.pkl".format(str(args.split)), "rb") as f:
    Xtrain = pickle.load(f)
print(Xtrain.head(), flush=True)
with open("ST258_Imi_CoreAndAGEs_RF_f1Tuned_ytrain_split{0}.pkl".format(str(args.split)), "rb") as f:
    ytrain = pickle.load(f)
print(ytrain.head(), flush=True)

print("Test set:", flush=True)
with open("ST258_Imi_CoreAndAGEs_RF_f1Tuned_Xtest_split{0}.pkl".format(str(args.split)), "rb") as f:
    Xtest = pickle.load(f)
print(Xtrain.head(), flush=True)
with open("ST258_Imi_CoreAndAGEs_RF_f1Tuned_ytest_split{0}.pkl".format(str(args.split)), "rb") as f:
    ytest = pickle.load(f)
print(ytest.head(), flush=True)

# Load in SNV and UG feature lists
print("Feature Lists:")
with open("ST258_Imi_SNV_features.pkl".format(str(args.split)), "rb") as f:
    SNV_features = pickle.load(f)
print(SNV_features[0:10], flush=True)
with open("ST258_Imi_UG_features.pkl".format(str(args.split)), "rb") as f:
    UG_features = pickle.load(f)
print(UG_features[0:10], flush=True)

# Set seed for repeatable results (based on previous call of np.random.randint(low=1, high=1e9))
s = 598807245
print("Seed used in this analysis is " + str(s), flush=True)

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

# Make transformer to do one-hot encoding on SNV_sites and leave AGEs alone
transformer = ColumnTransformer(
    [("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), SNV_features),
     ("pt", "passthrough", UG_features)
     ]
)

# Create pipeline to do both one-hot encoding and then RF
pipe_CG_AG = Pipeline(steps=[
    ("transformer", transformer),
    ("rf", RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True))
])
print("Pipeline:", flush=True)
print(pipe_CG_AG, flush=True)

# Create GSCV estimator
param_grid = {'rf__max_features': ['sqrt', 'log2'],
              'rf__min_samples_split': [2, 4],
              'rf__min_samples_leaf': [1, 2, 4, 6],
              'rf__criterion': ['gini', 'entropy'],
              'rf__max_depth': [None, 10, 20, 30],
              'rf__class_weight': ['balanced', 'balanced_subsample', None]}
GSCV_CG_AG = GridSearchCV(pipe_CG_AG, param_grid, scoring="f1", cv=stratkf, iid=False, n_jobs=-1)

# Fit estimator to training data for this fold
print("Fit estimator to training dataset and show best parameters")
GSCV_CG_AG.fit(Xtrain,ytrain)
print(GSCV_CG_AG, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal score: " + str(GSCV_CG_AG.best_score_), flush=True)

#Print best parameters
print("Best parameters:", flush=True)
print(GSCV_CG_AG.best_params_, flush=True)

print("Complete CV results:", flush=True)
print(GSCV_CG_AG.cv_results_, flush=True)

# Pickle model for fold
with open('ST258_Imi_CoreAndAGEs_RF_f1Tuned_split{0}_GridSearch.pkl'.format(str(args.split)), 'wb') as f:
    pickle.dump(GSCV_CG_AG, f)

# Test against CV test set for this fold
print("Test performance against test set for this fold", flush=True)
# Predict values for the testing data
ypred = GSCV_CG_AG.predict(Xtest)
y_pred_prob = GSCV_CG_AG.predict_proba(Xtest)[:, 1]
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
Scores.to_csv("ST258_Imi_CoreAndAGEs_RF_f1Tuned_split{0}_CVResults.csv".format(str(args.split)), index=False)
