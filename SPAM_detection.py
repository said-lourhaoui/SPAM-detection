#===============================================================================
# EE5907 : Pattern recognition
# Class assignment 1
# Monday 21,Sep 2020
#
# student : Said Lourhaoui
# email : e0572544@u.nus.edu
#===============================================================================
# README
# This project implements different algorithms to detect whether an email is a spam or not.
#
# * Place the script and the data file (spamData.mat) in the same directory.
# * To be able to run the code the following modules are required : scipy, numpy, matplotlib and sklearn.
# * You can install them using the following command, on cmd or shell :
# >>> python -m pip install ModuleName
#
# * Use a python3 interpreter to run the code
#===============================================================================
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances  # used to accelerate euclidean distances calculations

####################### Data loading ###########################################
annots = loadmat('spamData.mat')
# print(annots)
Xtrain = annots["Xtrain"]
Xtest  = annots["Xtest"]
ytrain = annots["ytrain"]
ytest  = annots["ytest"]


print("Shape of \n Xtrain : {} \n Xtest  : {} \n ytrain : {} \n ytest  : {}".format(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape))
Nrows, Ncolumns = Xtrain.shape
Nrows_test, Ncolumns_test = Xtest.shape
####################### Data Processing ########################################
### log-transform ###
def log_likelihood(ftr_array):
    """
    This function takes as input an array of features,
    and outputs the log-likelihood of the features in a new array
    Parameters :
    ftr_array : The features array
    """
    rows, columns = ftr_array.shape
    log_array = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            log_array[i][j] = np.log(ftr_array[i][j]+0.1)
    return log_array
### binarization  ###
def binarize(ftr_array):
    """
    This function takes as input an array of features, and binarizes them,
    i.e if a features is greater than 0, it is set to 1. Otherwise it is set to 0.
    Parameters :
    ftr_array : The features array
    """
    rows, columns = ftr_array.shape
    bin_array = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            bin_array[i][j] = 1 if ftr_array[i][j] > 0 else 0
    return bin_array

### Preprocessing of the features using binarization
Xtrain_bin = binarize(Xtrain)
Xtest_bin  = binarize(Xtest)

### Preprocessing of the features using log-likelihood
Xtrain_log = log_likelihood(Xtrain)
Xtest_log  = log_likelihood(Xtest)


########################## Q1 : Beta-binomial Naive Bayes ######################
print("===========Beta-binomial Naive Bayes=========")
### Class label prior ###
Lambda_ML = np.sum(ytrain)/(ytrain.shape[0])

### The posterior predective distribution ###
def theta_jc(alpha, X, Y):
    """
    This function computes the posterior predective distribution for each class ( y = 1 :Spam and y = 0 : Not spam)
    Theta_jc = (Njc+alpha)/(Nc+alpha)
    It outputs an array of two rows (class 0 and 1) and 57 columns (features 1-57)
    """
    rows, columns = X.shape
    Theta_array = [[], []]
    # fill the first row corresponding to c = 0
    for j in range(columns):
        N0, Nj0 = 0, 0
        N1, Nj1 = 0, 0
        for row in range(rows):
            if Y[row,0]==0:  # c = 0
                N0+=1
                if X[row,j]==1:
                    Nj0+=1
            else :   # Y[row,0]==1: # c = 1
                N1+=1
                if X[row,j]==1:
                    Nj1+=1
        # fill the first row corresponding to c = 0
        Theta_array[0] = np.append(Theta_array[0], (Nj0+alpha)/(N0+2*alpha))
        # fill the second row corresponding to c = 1
        Theta_array[1] = np.append(Theta_array[1], (Nj1+alpha)/(N1+2*alpha))

    return Theta_array

### fitting the model on the training set ###

alpha_array = np.linspace(0,100,201)
train_error_array = []
print("========= Training =========")
print("alpha : ")
for alpha in alpha_array:
    print(alpha)
    Theta_jc = theta_jc(alpha, Xtrain_bin, ytrain)
    predicted_Y = np.zeros((Nrows,1))
    train_error = 0
    log_Pyx_0 = []   #This list will contain the log(p(y=0|x,D))
    log_Pyx_1 = []   #This list will contain the log(p(y=1|x,D))
    for i, current_row in enumerate(Xtrain_bin):
        log_Pxy_0 = 0 # initialize the sum of log(P(x |y,theta_jc))  , y= 0
        log_Pxy_1 = 0 # initialize the sum of log(P(x |y,theta_jc))  , y= 1

        for j, current_column in enumerate(current_row):
            #current_column = Xtrain_bin[i,j] =xij
            # log_Pxy += np.log((Theta_jc[0][j]**xij)*(1-Theta_jc[0][j])**(1-xij)) # if xij == 1 we get log(thetajc) else we get log(1-thethajc)
            if current_column == 1:
                log_Pxy_0 += np.log(Theta_jc[0][j])
                log_Pxy_1 += np.log(Theta_jc[1][j])
            else :
                log_Pxy_0 += np.log(1-Theta_jc[0][j])
                log_Pxy_1 += np.log(1-Theta_jc[1][j])
        ### log(p(y=0|x,D))
        log_Pyx_0.append(np.log(1-Lambda_ML)+log_Pxy_0)
        ### log(p(y=1|x,D))
        log_Pyx_1.append(np.log(Lambda_ML)+log_Pxy_1)


    ###  Get the predicted Y and store in the class vector
        predicted_Y[i,0] = 1 if log_Pyx_1[i]>log_Pyx_0[i] else 0

    ### Calculate error for the current sample
        train_error += 1 if predicted_Y[i,0] != ytrain[i,0] else 0
    ### Calculate the error after training
    train_error_rate = train_error / Nrows

    train_error_array = np.append(train_error_array, [train_error_rate])

    if alpha==1:   # 0.10962479608482871
        print("Training error for alpha = 1 :", train_error_rate)
    if alpha==10:  # 0.11582381729200653
        print("Training error for alpha = 10 :", train_error_rate)
    if alpha==100: # 0.13605220228384993
        print("Training error for alpha = 100 :", train_error_rate)


### Testing the model on the test set ###

alpha_array = np.linspace(0,100,201)
test_error_array = []
print("========= Testing =========")
print("alpha : ")
for alpha in alpha_array:
    print(alpha)
    Theta_jc = theta_jc(alpha, Xtrain_bin, ytrain)
    predicted_Y_test = np.zeros((Nrows_test,1))
    test_error = 0
    log_Pyx_0 = []  #This list will contain the log(p(y=0|x,D))
    log_Pyx_1 = []  #This list will contain the log(p(y=1|x,D))
    for i, current_row in enumerate(Xtest_bin):
        log_Pxy_0 = 0   # initialize the sum of log(P(x |y,theta_jc)) , y = 0
        log_Pxy_1 = 0   # initialize the sum of log(P(x |y,theta_jc))  , y = 1
        for j, current_column in enumerate(current_row):
            #current_column = Xtest_bin[i,j] = xij
            #log_Pxy += np.log((Theta_jc[0][j]**xij)*(1-Theta_jc[0][j])**(1-xij)) # if xij == 1 we get log(thetajc) else we get log(1-thethajc)
            if current_column == 1:
                log_Pxy_0 += np.log(Theta_jc[0][j])
                log_Pxy_1 += np.log(Theta_jc[1][j])
            else :
                log_Pxy_0 += np.log(1-Theta_jc[0][j])
                log_Pxy_1 += np.log(1-Theta_jc[1][j])
        ### log(p(y=0|x,D))
        log_Pyx_0.append(np.log(1-Lambda_ML)+log_Pxy_0)
        ### log(p(y=1|x,D))
        log_Pyx_1.append(np.log(Lambda_ML)+log_Pxy_1)

    ###  Get the predicted Y and store in the class vector
        predicted_Y_test[i,0] = 1 if log_Pyx_1[i]>log_Pyx_0[i] else 0


    ### Calculating the error of the current sample
        test_error += 1 if predicted_Y_test[i,0] != ytest[i,0] else 0
    test_error_rate = test_error / Nrows_test

    test_error_array = np.append(test_error_array, [test_error_rate])

    if alpha==1:  # 0.11393229166666667
        print("Test error for alpha = 1 : ", test_error_rate)
    if alpha==10: # 0.12434895833333333
        print("Test error for alpha = 10 :", test_error_rate)
    if alpha==100: # 0.14583333333333334
        print("Test error for alpha = 100 :", test_error_rate)

plt.figure(1)
plt.plot(alpha_array, train_error_array, color='g', label="Training error rate")
plt.plot(alpha_array, test_error_array, color='r', label="Test error rate")
plt.title("Error rates for Beta-Binomial Naive Bayes Classifier")
plt.legend()
plt.xlabel("alpha")
plt.ylabel("error rate")
plt.show()

######################### Q2 Gaussian Naive Bayes ##############################
print("===========Gaussian Naive Bayes=========")
def mean_jc(X,Y):
    rows, columns = X.shape
    mean_jc_array = np.zeros((2,57))
    # fill the first row corresponding to c = 0
    for j in range(columns):
        N0, Xj0 = 0, 0
        N1, Xj1 = 0, 0
        for row in range(rows):
            if Y[row,0]==0:  # c = 0
                N0+=1
                Xj0+= X[row,j]
            else :  # c = 1
                N1+=1
                Xj1+= X[row,j]
        # fill the second row corresponding to c = 0
        mean_jc_array[0,j]= Xj0 / N0
        # fill the second row corresponding to c = 1
        mean_jc_array[1,j]= Xj1 / N1

    return mean_jc_array
### compute the mean
mean = mean_jc(Xtrain_log,ytrain)


def var_jc(X,Y):
    rows, columns = X.shape
    var_jc_array = np.zeros((2,57))
    # fill the first row corresponding to c = 0
    for j in range(columns):
        N0, Xj0 = 0, 0
        N1, Xj1 = 0, 0
        for row in range(rows):
            if Y[row,0]==0:  # c = 0
                N0+=1
                Xj0+= (X[row,j]-mean[0,j])**2
            else : # c = 1
                N1+=1
                Xj1+= (X[row,j]-mean[1,j])**2
        # fill the second row corresponding to c = 0
        var_jc_array[0,j]= Xj0 / N0
        # fill the second row corresponding to c = 1
        var_jc_array[1,j]= Xj1 / N1

    return var_jc_array
### compute the variance
var = var_jc(Xtrain_log, ytrain)

### fitting the model on the training set log-likelihood ###
predicted_Y_train_l = np.zeros((Nrows,1))
log_Pyx_train_l = np.zeros((Nrows,2))
train_error_l = 0
### Compute log(p(y=0|x,lambda,(mu,sigma))) and log(p(y=1|x,lambda,(mu,sigma)))
for i in range(Nrows):
    log_Pxy_train_l_0 = 0
    log_Pxy_train_l_1 = 0
    for j in range(Ncolumns):
        log_Pxy_train_l_0 += -0.5*np.log(2*np.pi*var[0,j])-0.5*((Xtrain_log[i,j]-mean[0,j])**2)/var[0,j]
        log_Pxy_train_l_1 += -0.5*np.log(2*np.pi*var[1,j])-0.5*((Xtrain_log[i,j]-mean[1,j])**2)/var[1,j]

    log_Pyx_train_l[i,0] += np.log(1-Lambda_ML) + log_Pxy_train_l_0  # log(p(y=0|x,lambda,(mu,sigma)))
    log_Pyx_train_l[i,1] += np.log(Lambda_ML) + log_Pxy_train_l_1    # log(p(y=1|x,lambda,(mu,sigma)))

### Classification
    predicted_Y_train_l[i,0] = 1 if log_Pyx_train_l[i,1]>log_Pyx_train_l[i,0] else 0

### Calculating the training error
    train_error_l += 1 if predicted_Y_train_l[i] != ytrain[i] else 0
train_error_rate_l = train_error_l / Nrows
print("Training error rate for Gaussian Naive Bayes : ", train_error_rate_l)

### testing the model on the test set log-likelihood ###
predicted_Y_test_l = np.zeros((Nrows_test,1))
log_Pyx_test_l = np.zeros((Nrows_test,2))
test_error_l = 0
### compute log(p(y=0|x,lambda,(mu,sigma)))  and log(p(y=1|x,lambda,(mu,sigma)))
for i in range(Nrows_test):
    log_Pxy_test_l_0 = 0
    log_Pxy_test_l_1 = 0
    for j in range(Ncolumns):
        log_Pxy_test_l_0 += -0.5*np.log(2*np.pi*var[0,j])-0.5*((Xtest_log[i,j]-mean[0,j])**2)/var[0,j]
        log_Pxy_test_l_1 += -0.5*np.log(2*np.pi*var[1,j])-0.5*((Xtest_log[i,j]-mean[1,j])**2)/var[1,j]

    log_Pyx_test_l[i,0] += np.log(1-Lambda_ML) + log_Pxy_test_l_0  # log(p(y=0|x,lambda,(mu,sigma)))
    log_Pyx_test_l[i,1] += np.log(Lambda_ML) + log_Pxy_test_l_1    # log(p(y=1|x,lambda,(mu,sigma)))

### Classification
    predicted_Y_test_l[i,0] = 1 if log_Pyx_test_l[i,1]>log_Pyx_test_l[i,0] else 0

### calculating the testing error
    test_error_l += 1 if predicted_Y_test_l[i] != ytest[i] else 0
test_error_rate_l = test_error_l / Nrows_test
print("Test error rate for Gaussian Naive Bayes : ",  test_error_rate_l)

################### Q3 Logistic Regression ########################################
print("===========Logistic Regression=========")
### Add one to feature vectors
Xtrain_bias  = np.insert(Xtrain_log, 0, 1, axis = 1)
Xtest_bias   = np.insert(Xtest_log, 0, 1, axis = 1)
lambda_vec   = np.append(np.arange(1,11,1), np.arange(15, 101, 5))
step_size    = 1
train_error_array = []
test_error_array  = []
### fitting the model on the training set of log-likelihood data #####################

# print("Lambda :")
for Lambda in lambda_vec :
    print(Lambda)
    w = np.zeros((Ncolumns+1))
    I = np.eye(Ncolumns+1)
    I[0,0] = 0
    S = np.eye(Nrows)          # NxN diagonal matrix with i-th is ui*(1-ui)
    err = 10
    treshold = 10**(-5)
    mu = np.zeros((Nrows,1))   # Dx1 vector
    while err > treshold:
        # gw = np.zeros((Ncolumns+1, 1))  # (D+1)x1 vector
        # hw = 0
        for i in range(Nrows):
            xi = Xtrain_bias[i,:]
            mu_i = 1/(1+np.exp(-w.T.dot(xi))) # sigmoid function
            mu[i,0] = mu_i
            S[i,i] = mu_i*(1-mu_i)
            # gw[j,0] += (mu_i - ytrain[i,0])*xi.T
        # print(gw)

        greg = Xtrain_bias.T.dot(mu[:,0]-ytrain[:,0]) + Lambda*(np.append([0],w[1:,]))
        Hreg = Xtrain_bias.T.dot(S).dot(Xtrain_bias) + Lambda*I
        # print(greg)
        # print(Hreg)
        try : # Try to compute the inverse of Hreg
            Hreg_inv = np.linalg.inv(Hreg)
        except np.linalg.linalg.LinAlgError : # Otherwise compute the pseudo inverse
            Hreg_inv = np.linalg.pinv(Hreg)
        dk = Hreg_inv.dot(greg)   # descent direction
        w = w - step_size*dk

        err = dk.T.dot(dk)

### Classification  : 1 spam, 0 non-spam
    log_p_train = [w.T.dot(Xtrain_bias[i]) for i in range(Nrows)]
    predicted_Y_train = np.array([1 if log_p_train[i]>0 else 0 for i in range(Nrows)])

    log_p_test = [w.T.dot(Xtest_bias[i]) for i in range(Nrows_test)]
    predicted_Y_test = np.array([1 if log_p_test[i]>0 else 0 for i in range(Nrows_test)])
### Calculate training error rate
    train_error_l = 0
    for i in range(Nrows):
        train_error_l += 1 if predicted_Y_train[i] != ytrain[i] else 0
    train_error_rate_l = train_error_l/Nrows
    train_error_array = np.append(train_error_array, [train_error_rate_l])
### Calculate test error rate
    test_error_l = 0
    for i in range(Nrows_test):
        test_error_l += 1 if predicted_Y_test[i] != ytest[i] else 0
    test_error_rate_l = test_error_l/Nrows_test
    test_error_array = np.append(test_error_array, [test_error_rate_l])

    if Lambda == 1:
        print("Train error rate for lambda = 1 : ", train_error_rate_l) #0.04926590538336052
        print("Test error rate for lambda = 1 : ", test_error_rate_l)   #0.061848958333333336
    if Lambda == 10:
        print("Train error rate for lambda = 10 : ", train_error_rate_l) #0.05187601957585644
        print("Test error rate for lambda = 10 : ", test_error_rate_l)   #0.061197916666666664
    if Lambda == 100:
        print("Train error rate for lambda = 100 : ", train_error_rate_l)  #0.06133768352365416
        print("Test error rate for lambda = 100 : ", test_error_rate_l)    #0.06901041666666667

plt.figure(2)
plt.plot(lambda_vec, train_error_array, color='g', label="Training error rate")
plt.plot(lambda_vec, test_error_array, color='r', label="Test error rate")
plt.title("Error rates for Logistic Regression")
plt.legend()
plt.xlabel("lambda")
plt.ylabel("error rate")
plt.show()

############### K-Nearest Neighbors ############################################
print("===========K-Nearest Neighbors=========")
K_vec = np.append(np.arange(1,11,1), np.arange(15, 101, 5))
test_error_array = []
train_error_array = []
### Calculate euclidien distance between two vectors
def euclidean_dist(vec1, vec2):
    """
    This function calculates the euclidien distance between two vectores
    Make sure vec1 and vec2 are numpy arrays
    """
    distance = np.sqrt(np.sum((vec1-vec2)**2))
    return distance

### fitting the model on the training set

# distances_1 = np.zeros((Nrows, Nrows))
# for train_row_1 in range(Nrows):
#     for train_row_2 in range(Nrows):
#         if train_row_1 != train_row_2 :
#             # distances_1[train_row_1,train_row_2] = euclidean_dist(Xtrain_log[train_row_1,:], Xtrain_log[train_row_2,:])
#             distances_1[train_row_1,train_row_2] = np.sqrt(np.dot(Xtrain_log[train_row_1,:], Xtrain_log[train_row_1,:])-2*np.dot(Xtrain_log[train_row_1,:], Xtrain_log[train_row_2,:])+ np.dot(Xtrain_log[train_row_2,:],Xtrain_log[train_row_2,:]) )

distances = euclidean_distances(Xtrain_log, Xtrain_log)  ## we use sklearn euclidean distance to accelerate the computation

for k in K_vec:
    # print(k)
    predicted_Y_train_knn = []
    train_error_knn = 0
    for train_row in range(Nrows):
        k_neighbors_train = [] # It contains the class (0 or 1 of the k neighbors)
        ## sort
        indexes = np.argsort(distances[train_row,:])
        for e in range(k):
            # Add the k nearest neighbors to the list
            k_neighbors_train.append(ytrain[indexes[e]])

        p_1 = sum(k_neighbors_train)/k
        p_0 = 1 - p_1

        predicted_Y_train_knn = np.append(predicted_Y_train_knn, [1 if p_1>p_0 else 0])

### Calculate training error rate
        train_error_knn += 1 if predicted_Y_train_knn[train_row] != ytrain[train_row] else 0
    train_error_rate_knn = train_error_knn/Nrows
    train_error_array = np.append(train_error_array, [train_error_rate_knn])

    if k == 1:
        print("Train error rate for k = 1 : ", train_error_rate_knn)
    if k == 10:
        print("Train error rate for k = 10 : ", train_error_rate_knn)
    if k == 100:
        print("Train error rate for k = 100 : ", train_error_rate_knn)

# distances = np.zeros((Nrows_test, Nrows))
# for test_row in range(Nrows_test):
#     for train_row in range(Nrows):
#         distances[test_row,train_row] = euclidean_dist(Xtest_log[test_row,:], Xtrain_log[train_row,:])
distances = euclidean_distances(Xtest_log, Xtrain_log)

for k in K_vec:
    print(k)
### Predicting Y for the test set
    predicted_Y_test_knn = []
    test_error_knn = 0
    for test_row in range(Nrows_test):
        # print("test_row",test_row)
        ## sort the current row of the distances
        k_neighbors_test = []
        indexes = np.argsort(distances[test_row,:])
        for e in range(k):
            # Add the k nearest neighbors to the list
            k_neighbors_test.append(ytrain[indexes[e]])

        p_1 = sum(k_neighbors_test)/k  # k_1/k
        p_0 = 1 - p_1

        predicted_Y_test_knn = np.append(predicted_Y_test_knn, [1 if p_1>p_0 else 0])
### Calculate testing error rate
        test_error_knn += 1 if predicted_Y_test_knn[test_row] != ytest[test_row] else 0
    test_error_rate_knn = test_error_knn/Nrows_test
    test_error_array = np.append(test_error_array, [test_error_rate_knn])

    if k == 1:
        print("Test error rate for k = 1 : ", test_error_rate_knn)
    if k == 10:
        print("Test error rate for k = 10 : ", test_error_rate_knn)
    if k == 100:
        print("Test error rate for k = 100 : ", test_error_rate_knn)

plt.figure(3)
plt.plot(K_vec, train_error_array, color='g', label="Training error rate")
plt.plot(K_vec, test_error_array, color='r', label="Test error rate")
plt.title("Error rates for K-Nearest Neighbors")
plt.legend()
plt.xlabel("K")
plt.ylabel("error rate")
plt.show()
