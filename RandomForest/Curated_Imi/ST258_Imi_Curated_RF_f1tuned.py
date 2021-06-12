# ### Load in modules needed for data inport/proccessing, analysis, and plotting

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

#For plotting
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

#For plotting learning curves
import sys
sys.path.append('/path/to/python_functions/')
import plot_learning_curves_function_95CI

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
s = 116579641
print("Seed used in this analysis is " + str(s), flush=True)

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

# Definine scorer for nested cross-validation
scorer = {
    "accuracy": "accuracy",
    "sensitivity": "recall",
    "specificity": metrics.make_scorer(metrics.recall_score, greater_is_better=True, pos_label=0),
    "PPV": "precision",
    "AUC": "roc_auc",
    "f1": "f1"
}
print("Scoring strategy for CV:", flush=True)
print(scorer, flush=True)
print("", flush=True)

# Define X and y
Curated["rank"] = labels["imipenem_gt8"]

# Define X as all features and y as all labels
X = Curated.drop("rank", axis=1)
y = Curated["rank"]
print("Features:", flush=True)
print(X.head(), flush=True)
print("Labels:", flush=True)
print(y, flush=True)

# Instantiate RF estimator
RF_model = RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True)

# Create GSCV estimator
param_grid = {'max_features': ['sqrt', 'log2'],
              'min_samples_split': [2, 4],
              'min_samples_leaf': [1, 2, 4, 6],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 10, 20, 30],
              'class_weight': ['balanced', 'balanced_subsample', None]}
GSCV_Curated = GridSearchCV(RF_model, param_grid, scoring="f1", cv=stratkf, iid=False, n_jobs=-1)

# Fit estimator to all data and show best params
print("Fit estimator to complete dataset and show best parameters")
GSCV_Curated.fit(X,y)
print(GSCV_Curated, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal score: " + str(GSCV_Curated.best_score_), flush=True)

#Print best parameters and complete CV results
print("Best parameters:", flush=True)
print(GSCV_Curated.best_params_, flush=True)

print("Complete CV results:", flush=True)
print(GSCV_Curated.cv_results_, flush=True)

# Pickle final model
with open('ST258_Imi_Curated_RF_f1Tuned_GridSearch.pkl', 'wb') as f:
    pickle.dump(GSCV_Curated, f)

# Plot learning curve
GSCV_Curated = GridSearchCV(RF_model, param_grid, scoring="f1", cv=stratkf, iid=False, n_jobs=-1)

file_lc = "ST258_Imi_Curated_RF_f1Tuned_LearningCurve_95CI.pdf"
plot_learning_curves_function_95CI.plot_learning_curve(
    estimator=GSCV_Curated, X=X, y=y, scoring="f1", cv=stratkf, n_jobs=1,
    rs=s, ylab="F1 Score", ylim=(0,1.01))
plt.savefig(file_lc)
