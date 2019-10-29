print(__doc__)

###############################     INSTALLATION/PREP     ################
# This is a DEMO to demonstrate the classifiers we learned about
# in CSI 431 @ UAlbany Fall'18
#
# Might need to install the latest scikit-learn
# On linux or Mac: sudo pip install -U scikit-learn
#
# Codebase with more classifiers here:
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

###############################     IMPORTS     ##########################

# numeric python and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# utility to help you split training data
from sklearn.model_selection import train_test_split
# utility to standardize data http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler
# some dataset generation utilities. for example: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
from sklearn.datasets import make_classification

# Scoring for classifiers
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn.model_selection import cross_val_score

# Classifiers from scikit-learn
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# DT
from sklearn.tree import DecisionTreeClassifier
# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
#Extra Credit - Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder

###############################     CLASSIFIERS     #######################

# Put the names of the classifiers in an array to plot them
names = ["LDA",
         "Lin. SVM\nC=1",
         "DT\nGINI(20)",
         "DT\nIG(20)",
         #Extra Credit
         "RF\nGINI(20)",
         "RF\nIG(20)"
         ]

#Make Dummy Arrays to append f-measure to / C, k Values to loop through for plotting
svm_score = []
dt_gini_score = []
dt_ig_score = []
C = [0.01, 0.1, 1, 10, 100]
k = [2, 5, 10, 20]

# Create the classifiers with respective parameters
# LDA, NB: No parameters
# SVM:     One linear and with C=0.025 and one RBF kernel-SVM with C=1
# DT :     Limit depth to 5 (i.e. at most 5 consecutive splits in each decision rule)

best_Classifiers = [
    LinearDiscriminantAnalysis(),
    SVC(kernel="linear", C=1),
    #Best GINI && #Best IG (not accurate yet)
    DecisionTreeClassifier(max_leaf_nodes=2),
    DecisionTreeClassifier(max_leaf_nodes=2, criterion = "entropy"),
    #Extra Credit
    RandomForestClassifier(max_leaf_nodes=2),
    RandomForestClassifier(max_leaf_nodes=2, criterion = "entropy"),
]

###############################     DATASETS     ##########################

#Load train class/data
trainData = np.loadtxt("cancer-data-train.csv", delimiter=',', usecols = range(0,30), dtype=np.float)
trainClass = np.loadtxt("cancer-data-train.csv", delimiter=',', dtype=str) [:,-1]

#Reshape data
trainClass = trainClass.reshape(-1,1)
encoder = OrdinalEncoder()
trainClass = encoder.fit_transform(trainClass).flatten()    #converts M/B into 0s and 1s

#Load test class/data
testData = np.loadtxt("cancer-data-test.csv", delimiter=',', usecols = range(0,30), dtype=np.float)
testClass = np.loadtxt("cancer-data-test.csv", delimiter=',', dtype=str) [:,-1]
testClass = testClass.reshape(-1,1)
testClass = encoder.fit_transform(testClass).flatten()    #converts M/B into 0s and 1s



data = (trainData,trainClass)

# put our datasets in an array
datasets = [
            data
            ]

###############################  TRAIN AND PLOT  ##########################

#Initialize Graphs
fig, (precision,recall,fscore) = plt.subplots(1,3)

# Iterate over datasets and train and plot each classifier
for ds_cnt, ds in enumerate(datasets):
    # Preprocess dataset, split into training and test part
    X, y = ds
    # Standardize
    X = StandardScaler().fit_transform(X)
    # Splits our dataset in training and testing: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.1, random_state=42)

    # iterate over best_Classifiers
    for name, clf in zip(names, best_Classifiers):
        # Train the classifier (all classifiers in Scikit implement this functions)
        clf.fit(X, y)

        # Predict
        y_pred = clf.predict(np.array(testData, dtype=np.float))
        ap = average_precision_score(testClass, y_pred)
        rec = recall_score(testClass, y_pred, average='weighted')
        f1 = f1_score(testClass, y_pred, average='weighted', labels=np.unique(y_pred))

        #Plots current Classifier in each bar
        precision.bar(name,ap)
        recall.bar(name,rec)
        fscore.bar(name,f1)

#Set titles
precision.set_title("Precision")
recall.set_title("Recall")
fscore.set_title("F-Score")
plt.show()

print(testData.shape)
print(testClass.shape)
print(trainData.shape)
print(trainClass.shape)
