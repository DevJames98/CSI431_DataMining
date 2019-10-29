import numpy as np
import sklearn

def IG(D, index, value):
    #Compute entrophy 1st
    #D[1] - D yes | D[2] - D no

    cy = [x[1] for x in D].count(0) #loops through each row in D (tuple) and returns the class (index 1)
    length = len(D) #gets number of tuples

    #Find Prob. of cy
    probY = cy/length
    #Find Prob. of cn
    probN = 1 - probY
    #Entropy Calculation/Formula
    ent = -1 * ((probY * np.log2(probY)) + (probN * np.log2(probN)))

    #Compute split entrophy 2nd
    #Find ny/n
    sdataY1 = []        #split data for Yes (nY)
    sdataN1 = []        #split data for No (nN)
    cdataY1 = []        #class data for Yes (nY)
    cdataN1 = []        #class data for No (nN)

    data = [x[0] for x in D]
    classI = [x[1] for x in D]
    data = np.atleast_2d(data)  #converts to at least 2d numpy arr

    #Splitting the data
    for i in range(0,length):               #loop through each row in D
        if data[i,index] > value:
            sdataY1.append(data[i])         #append data to new array
            cdataY1.append(classI[i])       #append class info to new array
        else:
            sdataN1.append(data[i])
            cdataN1.append(classI[i])

    nY = len(sdataY1)       # # of samples/things split into dY (Yes class)
    nN = len(sdataN1)       # # of samples/things split into dN (No class)
    n = len(data)           # Total # of samples in D (Total class)

    cYY = cdataY1.count(0)      #counts '0' classes in Yes class
    cYN = cdataY1.count(1)      #counts '1' classes in Yes class
    cNY = cdataN1.count(0)      #counts '0' classes in No class
    cNN = cdataN1.count(1)      #counts '1' classes in No class

    #to avoid divide by zero error, change P(0) to 1 (if a class has 0 prob.)
    if cYY == 0:
        cYY = 1
    else:
        cYY = cYY/n             #so that probability is calculated instead

    if cYN == 0:
        cYN = 1
    else:
        cYN = cYN/n             #so that probability is calculated instead

    if cNY == 0:
        cNY = 1
    else:
        cNY = cNY/n             #so that probability is calculated instead

    if cNN == 0:
        cNN = 1
    else:
        cNN = cNN/n             #so that probability is calculated instead


    #Split Entrophy Calculation/Formula
    #entrophy for cYY + cYN + cNY + cNN (1st letter - OG class, 2nd letter - subclass)
    se_cy = -1 * ((cYY)* np.log2(cYY) + ((cYN)* np.log2(cYN))) * (nY/n)         #entrophy * nY/n (prob)
    se_cn = -1 * ((cNY)* np.log2(cNY) + ((cNN)* np.log2(cNN))) * (nN/n)         #entrohpy * nN/n (prob)

    splitEnt = se_cy + se_cn

    #return Information Gain
    infGain = ent - splitEnt
    return infGain

"""Compute the Information Gain of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Information Gain for the given split
    """



def G(D, index, value):

    length = len(D) #gets number of tuples

    #Compute split Gini index

    sdataY1 = []        #split data for Yes (nY)
    sdataN1 = []        #split data for No (nN)
    cdataY1 = []        #class data for Yes (nY)
    cdataN1 = []        #class data for No (nN)

    data = [x[0] for x in D]
    classI = [x[1] for x in D]
    data = np.atleast_2d(data)  #converts to at least 2d numpy arr

    #Splitting the data
    for i in range(0,length):               #loop through each row in D
        if data[i,index] > value:
            sdataY1.append(data[i])         #append data to new array
            cdataY1.append(classI[i])       #append class info to new array
        else:
            sdataN1.append(data[i])
            cdataN1.append(classI[i])

    nY = len(sdataY1)       # # of samples/things split into dY (Yes class)
    nN = len(sdataN1)       # # of samples/things split into dN (No class)
    n = len(data)           # Total # of samples in D (Total class)

    cYY = cdataY1.count(0)      #counts '0' classes in Yes class
    cYN = cdataY1.count(1)      #counts '1' classes in Yes class
    cNY = cdataN1.count(0)      #counts '0' classes in No class
    cNN = cdataN1.count(1)      #counts '1' classes in No class

    ginicY = (nY/n) * (1 - ((np.square(cYY/n))+(np.square(cYN/n))))     #Yes class partition (Dy)
    ginicN = (nN/n) * (1 - ((np.square(cNY/n))+(np.square(cNN/n))))     #No class partition (Dn)

    #Gini Split index Calculation/Formula
    giniTot = ginicY + ginicN
    return giniTot

"""Compute the Gini index of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Gini index for the given split
    """


def CART(D, index, value):
    #Compute probabilities first
    #cy = [x[1] for x in D].count(0) #loops through each row in D (tuple) and returns the class (index 1)
    length = len(D) #gets number of tuples

    #Find Prob. of cy
    #probY = cy/length
    #Find Prob. of cn
    #probN = 1 - probY


    #Compute split of data
    #Find ny/n
    sdataY1 = []        #split data for Yes (nY)
    sdataN1 = []        #split data for No (nN)
    cdataY1 = []        #class data for Yes (nY)
    cdataN1 = []        #class data for No (nN)

    data = [x[0] for x in D]
    classI = [x[1] for x in D]
    data = np.atleast_2d(data)  #converts to at least 2d numpy arr

    #Splitting the data
    for i in range(0,length):               #loop through each row in D
        if data[i,index] > value:
            sdataY1.append(data[i])         #append data to new array
            cdataY1.append(classI[i])       #apped class info to new array
        else:
            sdataN1.append(data[i])
            cdataN1.append(classI[i])

    nY = len(sdataY1)       # # of samples/things split into dY (Yes class)
    nN = len(sdataN1)       # # of samples/things split into dN (No class)
    n = len(data)           # Total # of samples in D (Total class)

    cYY = cdataY1.count(0)      #counts '0' classes in Yes class
    cYN = cdataY1.count(1)      #counts '1' classes in Yes class
    cNY = cdataN1.count(0)      #counts '0' classes in No class
    cNN = cdataN1.count(1)      #counts '1' classes in No class

    cartY = np.absolute((cYY/n)-(cNY/n))        #Prob. of Yes in "Yes" split - Prob. of Yes in "No" Split
    cartN = np.absolute((cYN/n)-(cNN/n))        #Prob. of No in "Yes" split - Prob. of No in "No" Split

    #Cart Calculation/Formula
    cartTol = 2 * (nY/n) * (nN/n) * (cartY + cartN)
    return cartTol



"""Compute the CART measure of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the CART measure for the given split
    """


def bestSplit(D, criterion):

    #Used to store the tuple that best fits the criterion
    best_tuple = tuple()


    data = [x[0] for x in D]
    data = np.atleast_2d(data)

    a,b = 0,1       #a - Maximize criterion by finding the greatest values starting from 0 | b-  Minimize criterion by finding the least value starting from 1

    #Transpose data

    for row in data:                        #loops through each row in the data
        for col in range(0,len(data.T)):    #goes through each col in the range of the transposed data set
            #Evaluate criterion ("IG","GINI","CART")
            if criterion == G:         #Gini - minimize criterion
                #
                if b > criterion(D,col,row[col]):
                    b = criterion(D,col,row[col])
                    best_tuple = (col,row[col])         #stores index/value to use for the tuple that matches the criterion
            else:                           #IG/CART - maximize criterion
                #
                if a < criterion(D,col,row[col]):
                    a = criterion(D,col,row[col])
                    best_tuple = (col,row[col])         #stores value of the tuple that matches the criterion



    #return tuple
    return best_tuple


"""Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """

    #functions are first class objects in python, so let's refer to our desired criterion by a single name


def load(filename):

    y = np.loadtxt(filename, delimiter=',', dtype=float)
    return [(np.array(t[0:-1]),np.array(t[-1])) for t in y]



"""Loads filename as a dataset. Assumes the last column is classes, and
    observations are organized as rows.

    Args:
        filename: file to read

    Returns:
        A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
        where X[i] comes from the i-th row in filename; y is a list or ndarray of
        the classes of the observations, in the same order
    """



def classifyIG(train, test):

    #Predicted Class
    pred = []

    #Find the best split
    index, val = bestSplit(train,IG)        #index of col, val to split at


    data = [x[0] for x in test]
    data = np.atleast_2d(data)



    for x in range(0,len(data)):
        if data[x,index] > val:
            pred.append(1)         #append data to new array
        else:
            pred.append(0)


    return pred

"""Builds a single-split decision tree using the Information Gain criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """


def classifyG(train, test):
    #Predicted Class
    pred = []

    #Find the best split
    index, val = bestSplit(train,G)        #index of col, val to split at


    data = [x[0] for x in test]
    data = np.atleast_2d(data)



    for x in range(0,len(data)):
        if data[x,index] > val:
            pred.append(1)         #append data to new array
        else:
            pred.append(0)


    return pred

"""Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """


def classifyCART(train, test):
    #Predicted Class
    pred = []

    #Find the best split
    index, val = bestSplit(train,CART)        #index of col, val to split at


    data = [x[0] for x in test]
    data = np.atleast_2d(data)



    for x in range(0,len(data)):
        if data[x,index] > val:
            pred.append(1)         #append data to new array
        else:
            pred.append(0)


    return pred

"""Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """



def main():
    #Holds test and train data
    x = load("/Users/Devon/Desktop/School Stuff/uAlbany/Senior/Fall 2018/CSI 431/HW2/test.txt")
    y = load("/Users/Devon/Desktop/School Stuff/uAlbany/Senior/Fall 2018/CSI 431/HW2/train.txt")

    #Print Information Gain
    print("Information Gain: " + str(IG(x, 0, 1)))
    #Pring Gini Index
    print("Gini Index: " + str(G(x, 0, 1)))
    #Print CART
    print("CART: " + str(CART(x, 0, 1)))
    #Print Best Splits
    print("**Best Split**\nIG: " + str(bestSplit(y,IG)))
    print("G: " + str(bestSplit(y,G)))
    print("CART: " + str(bestSplit(y,CART)))

    #Print predicted classes
    print("Classify IG: " + str(classifyIG(y,x)))
    print("Classify G: " + str(classifyG(y,x)))
    print("Classify CART: " + str(classifyCART(y,x)))

    #Print out test class
    testClass = [z[1] for z in x]
    testClass = np.atleast_2d(testClass)
    print("Test Class: " + str(testClass))



"""This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point
    unlike C, Java, etc.
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
"""




if __name__=="__main__":
    main()

"""__name__=="__main__" when the python script is run directly, not when it
    is imported. When this program is run from the command line (or an IDE), the
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
"""
