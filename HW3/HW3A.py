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

# utility to help you split training data
from sklearn.model_selection import train_test_split
# utility to standardize data http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler
# some dataset generation utilities. for example: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score
# from sklearn.decomposition import PCA

# Classifiers from scikit-learn
# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder

###############################     CLASSIFIERS     #######################

# Put the names of the classifiers in an array to plot them
names = ["Linear SVM - C = .01",
         "Linear SVM - C = .1",
         "Linear SVM - C = 1",
         "Linear SVM - C = 10",
         "Linear SVM - C = 100"

         ]
#Make Dummy Arrays to append f-measure to / C Values to loop through for plotting
svm_score = []
C = [0.01, 0.1, 1, 10, 100]

# Create the classifiers with respective parameters
# SVM:     One linear and with C=0.025 and one RBF kernel-SVM with C=1
classifiers = [
    SVC(kernel="linear", C=0.01),
    SVC(kernel="linear", C=0.1),
    SVC(kernel="linear", C=1),
    SVC(kernel="linear", C=10),
    SVC(kernel="linear", C=100)
    ]


###############################     DATASETS     ##########################

# prepare a linearly separable dataset http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                          # random_state=1, n_clusters_per_class=1)

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

        #Calculate cross val score to find f measure
        score = cross_val_score(clf, X_test, y_test, cv = 10, scoring = 'f1_macro')

        #Appends cross validated f-measure to corresponding classifier array
        svm_score.append(np.mean(score))


#Part A
print("SVM Score")
print(svm_score)
#Plot results
plt.plot(C, svm_score, label="SVM")
plt.legend(loc='lower right')
plt.title('SVM Plot Graph')
plt.xlabel('C')
plt.ylabel('Avg. F measure')
plt.show()
