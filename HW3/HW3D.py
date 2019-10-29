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


from sklearn.model_selection import cross_val_score

# Classifiers from scikit-learn
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import OrdinalEncoder

###############################     CLASSIFIERS     #######################

# Put the names of the classifiers in an array to plot them
names = [
         "Random Forest\nGINI (2)",
         "Random Forest\nIG (2)",
         "Random Forest\nGINI (5)",
         "Random Forest\nIG (5)",
         "Random Forest\nGINI (10)",
         "Random Forest\nIG (10)",
         "Random Forest\nGINI (20)",
         "Random Forest\nIG (20)"
         ]

#Make Dummy Arrays to append f-measure to / k Values to loop through for plotting
rf_gini_score = []
rf_ig_score = []
k = [2, 5, 10, 20]

# Create the classifiers with respective parameters
# DT :     Limit depth to 5 (i.e. at most 5 consecutive splits in each decision rule)
classifiers = [
    RandomForestClassifier(max_leaf_nodes=2),
    RandomForestClassifier(max_leaf_nodes=2, criterion = "entropy"),
    RandomForestClassifier(max_leaf_nodes=5),
    RandomForestClassifier(max_leaf_nodes=5, criterion = "entropy"),
    RandomForestClassifier(max_leaf_nodes=10),
    RandomForestClassifier(max_leaf_nodes=10, criterion = "entropy"),
    RandomForestClassifier(max_leaf_nodes=20),
    RandomForestClassifier(max_leaf_nodes=20, criterion = "entropy")
    ]

###############################     DATASETS     ##########################

#Load train class/data
trainData = np.loadtxt("cancer-data-train.csv", delimiter=',', usecols = range(0,30), dtype=np.float)
trainClass = np.loadtxt("cancer-data-train.csv", delimiter=',', dtype=str) [:,-1]

#Reshape data
trainClass = trainClass.reshape(-1,1)
encoder = OrdinalEncoder()
trainClass = encoder.fit_transform(trainClass).flatten()    #converts M/B into 0s and 1s


data = (trainData,trainClass)

# put our datasets in an array
datasets = [
            #linearly_separable,
            data
            ]

###############################  TRAIN AND PLOT  ##########################

# Iterate over datasets and train and plot each classifier
for ds_cnt, ds in enumerate(datasets):
    # Preprocess dataset, split into training and test part
    X, y = ds
    # Standardize
    X = StandardScaler().fit_transform(X)
    # Splits our dataset in training and testing: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.1, random_state=42)

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        # Train the classifier (all classifiers in Scikit implement this functions)
        clf.fit(X_train, y_train)
        # Predict
        y_pred = clf.predict(X_test)


        #CLASSIFIER DETERMINANT
        score = cross_val_score(clf, X_test, y_test, cv = 10, scoring = 'f1_macro')

        #Appends cross validated f-measure to corresponding classifier array
        if "GINI" in name:
            rf_gini_score.append(np.mean(score))
        elif "IG" in name:
            rf_ig_score.append(np.mean(score))


#Part D
print("Decision Tree Score")
print(rf_gini_score)
print(rf_ig_score)
#Plot data
plt.plot(k, rf_gini_score, label="GINI")
plt.plot(k, rf_ig_score, label="IG")
plt.legend(loc='upper right')
plt.title('Random Forest Plot Graph')
plt.xlabel('k')
plt.ylabel('Avg. F measure')
plt.show()
