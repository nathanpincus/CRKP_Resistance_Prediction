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

# ### Import Core Genome SNVs, AGEs, and labels

# Read in fasta file
fasta = open("/path/to/CRE_ST258_core95.replaced.filtered.fasta").readlines()

# Extract out all genomes in alignment
genomes = []
for line in fasta:
    if line.startswith(">"):
        genomes.append(line.lstrip(">").rstrip("\n"))

# Add the sequence for each genome to a dictionary
alignment = {}
for line in fasta:
    for g in genomes:
        if line.rstrip("\n") == (">" + g):
            seqindex = fasta.index(line) + 1
            seq = fasta[seqindex].rstrip("\n")
            alignment[g] = [base for base in seq]

# Convert alignment dictionary to a pandas dataframe
alignDF = pd.DataFrame.from_dict(alignment, orient='index')
alignDF.index.name = "genome"
# Convert column names to strings by adding "x" to start avoid problems later
alignDF = alignDF.add_prefix("x")
SNV_features = alignDF.columns
print("Core genome dataset (head):", flush=True)
print(alignDF.head(), flush=True)
print("", flush=True)

# Import in AGE feature set, drop empty rank column, and save column names
UG = pd.read_csv("/path/to/ST258.uniquegroups.gte200.csv", index_col="genome")
UG = UG.drop("rank", axis=1)
UG = UG.astype("float64")
UG_features = UG.columns
print("Accessory genome dataset (head):", flush=True)
print(UG.head(), flush=True)
print("", flush=True)

# Import in labels.
# Genome name as index column
labels = pd.read_csv("/path/to/ST258_labels.csv", index_col="genome")
print("Labels:", flush=True)
print(labels.head(), flush=True)
print("", flush=True)

# Set seed for repeatable results (based on previous call of np.random.randint(low=1, high=1e9))
s = 139685890
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
CG_AG_Dataset = alignDF.join(UG, on="genome")
CG_AG_Dataset["rank"] = labels["meropenem_gt8"]

# Define X as all features and y as all labels
X = CG_AG_Dataset.drop("rank", axis=1)
y = CG_AG_Dataset["rank"]
print("Features:", flush=True)
print(X.head(), flush=True)
print("Labels:", flush=True)
print(y, flush=True)

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

# Fit estimator to all data and show best params
print("Fit estimator to complete dataset and show best parameters")
GSCV_CG_AG.fit(X,y)
print(GSCV_CG_AG, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal score: " + str(GSCV_CG_AG.best_score_), flush=True)

#Print best parameters and complete CV results
print("Best parameters:", flush=True)
print(GSCV_CG_AG.best_params_, flush=True)

print("Complete CV results:", flush=True)
print(GSCV_CG_AG.cv_results_, flush=True)

# Pickle final model
with open('ST258_Mero_CoreAndAGEs_RF_f1Tuned_GridSearch.pkl', 'wb') as f:
    pickle.dump(GSCV_CG_AG, f)

# Plot learning curve
GSCV_CG_AG = GridSearchCV(pipe_CG_AG, param_grid, scoring="f1", cv=stratkf, iid=False, n_jobs=-1)

file_lc = "ST258_Mero_CoreAndAGEs_RF_f1Tuned_LearningCurve_95CI.pdf"
plot_learning_curves_function_95CI.plot_learning_curve(
    estimator=GSCV_CG_AG, X=X, y=y, scoring="f1", cv=stratkf, n_jobs=1,
    rs=s, ylab="F1 Score", ylim=(0,1.01))
plt.savefig(file_lc)
