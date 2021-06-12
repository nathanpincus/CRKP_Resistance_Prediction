# Splitting dataset into folds for manual cross-validaiton

# Load in modules needed for data import/processing
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
# Save SNV_features to file
with open('ST258_Mero_SNV_features.pkl', 'wb') as f:
    pickle.dump(SNV_features, f)

# Import in AGE feature set, drop empty rank column, and save column names
UG = pd.read_csv("/path/to/ST258.uniquegroups.gte200.csv", index_col="genome")
UG = UG.drop("rank", axis=1)
UG = UG.astype("float64")
UG_features = UG.columns
print("Accessory genome dataset (head):", flush=True)
print(UG.head(), flush=True)
print("", flush=True)
# Save UG_features to file
with open('ST258_Mero_UG_features.pkl', 'wb') as f:
    pickle.dump(UG_features, f)

# Import in labels.
# Genome name as index column
labels = pd.read_csv("/path/to/ST258_labels.csv", index_col="genome")
print("Labels:", flush=True)
print(labels.head(), flush=True)
print("", flush=True)

# Set seed for repeatable results (based on previous call of np.random.randint(low=1, high=1e9))
s = 803084927
print("Seed used in this analysis is " + str(s), flush=True)

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

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
    with open('ST258_Mero_CoreAndAGEs_LR_f1Tuned_Xtrain_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(Xtrain, f)
    with open('ST258_Mero_CoreAndAGEs_LR_f1Tuned_ytrain_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(ytrain, f)
    with open('ST258_Mero_CoreAndAGEs_LR_f1Tuned_Xtest_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(Xtest, f)
    with open('ST258_Mero_CoreAndAGEs_LR_f1Tuned_ytest_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(ytest, f)
    split = split + 1
