import pandas as pd   
import numpy as np

# For visualisation
import matplotlib.pyplot as plt 
import seaborn as sns      
from mlxtend.plotting import plot_decision_regions  

# Models applied in this project
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC

# For results   
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

# For preprocessiing 
from random import randrange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV


############################################ VISUALISATION  AND PREPROCESSING ##################################################################

data = pd.read_csv('Iris.csv')  # read the iris dataset using panda 
print(data.describe())          # describe the dataset

# plot the different features of dataset
plt.title('PetalWidth')
plt.bar(data['Species'],data['PetalWidthCm'])
plt.show()
plt.title('PetalLength')
plt.bar(data['Species'],data['PetalLengthCm'])
plt.show()
plt.title('sepalWidth')
plt.bar(data['Species'],data['SepalWidthCm'])
plt.show()
plt.title('sepalLength')
plt.bar(data['Species'],data['SepalLengthCm'])
plt.show()


arr = np.array(data.values)        # make numpy arrays for easy calculations
x = np.zeros((arr.shape[0], arr.shape[1] - 2))  # considering all features 
x1 = arr[:,3:5]                      # considering only petal features
y = np.zeros((arr.shape[0],1))  # numpy array of labels
for i in range(arr.shape[0]):   
    x[i] = arr[i:i + 1, 1:5]
    t = arr[i:i+1, 5:]
    if t == "Iris-setosa":
        y[i] = '0'
    elif t=="Iris-versicolor":    # for our visualisation convert labels to 0,1,2
        y[i] = '1'
    else:
        y[i] = '2'
plt.plot(y)   # quantity of each label in dataset 
plt.show()


# for visualising purpose create train and test split using strified sampling
# scatter plot of all species w.r.t their features using seaborn 
data = data.drop(['Id'], axis=1)
train,test = train_test_split(data, test_size = 0.25, stratify = data['Species'], random_state = 0)
sns.pairplot(train, hue="Species")
plt.show()
# some variable seems to be highly correlated eg. petal length and petal width. petal measurements seperate the 
# species better then sepal ones.


# correlation matrix to see that correlation among variables
correlation_matrix = train.corr()
plt.figure(figsize = (8,6))
sns.heatmap(correlation_matrix, annot = True, square = True);
plt.show()
#the petal ones are highly positive correlated and on the other hand sepal ones are uncorrelated
#petal features has highly correlated with sepal length also but not with sepal width


######################################### LOGISTIC REGRESSION ##############################################################

print("######################## LOGISTIC REGRESSION ################################")
# Use stratified Sampling it divides the sample into subsamples and in each subsample randomly arrange the data
x_train, x_test, y_train, y_test = train_test_split(x,y.reshape(len(y),),stratify = y,test_size=0.25,random_state = 0)  # use 75:25 train-test split
sc = StandardScaler()
x_train = sc.fit_transform(x_train)    # use standard scalar for avoiding outliers 
x_test = sc.transform(x_test)

lr = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='multinomial',tol = .02) # we use lbfgs that use l2 penalties, to avoid overflow use tol
lr.fit(x_train,y_train.reshape(len(y_train),))
y_pred = lr.predict(x_test)

test_acc = accuracy_score(y_test,y_pred)
print("Test Accuracy using logistic regression:  ",test_acc)   # compute the test accuracy

y_train_pred = lr.predict(x_train)
print("Train Accuracy using logistic regression:  ",accuracy_score(y_train,y_train_pred)) # compute the train accuracy

c = confusion_matrix(y_test,y_pred)         # We want to see the confusion matrix of logistic model
plt.figure(figsize = (6,5))
sns.heatmap(c, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = plt.axes() ) # plot cofusion matrix using seaborn heatmap
plt.title('Confusion Matrix using logistic regression')
plt.show()

# cross_val_score used to avoid overfitting and estimate the skill of the model
Avg_acc = cross_val_score(lr, x, y.reshape(len(y)), cv = 5)               # apply k-cross fold to get stable results 
print("Average Accuracy using logistic regression and 5 folds:  ",np.mean(Avg_acc))

######################################### DECISION TREE ##############################################################

print("######################## DECISION TREE ######################################")
def grid_search(clf,param,x_train,y_train,x_test,y_test):  # perform grid search manually for decision tree and random forest
    maxx = 0                      # intialise the depth = 0
    for i in param["max_depth"]:   # param contains different values of depth from which we have to find best
        if clf == 'd':
            DT= DecisionTreeClassifier(max_depth= i)         # apply sklearn DT to get predicted values
        elif clf == 'r':
            DT=RandomForestClassifier(max_depth= i)
        DT.fit(x_train, y_train.reshape(len(y_train),))
        y_pred = DT.predict(x_test)
        s = accuracy_score(y_test,y_pred)   # calculate the accuracy
        if s>maxx:
            od = i
            maxx = s
    return od        # return the best depth possible 

x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(x,y.reshape(len(y),),test_size=0.25,random_state=0) # use 75:25 train-test split

param = {'max_depth':[2,3,4,5,None]}   # use some parameters for max_depth
od = grid_search("d",param,x_train_d,y_train_d,x_test_d,y_test_d)
print("Max depth: ",od)

dt= DecisionTreeClassifier(random_state = 0,max_depth = od)  # use sklearn DT model
dt.fit(x_train_d,y_train_d.reshape(len(y_train_d),))

y_pred_d = dt.predict(x_test_d)
test_acc_d = accuracy_score(y_test_d,y_pred_d)
print("Test accuracy using desision trees with optimal parameters: ",test_acc_d)  # compute the test accuracy

y_train_pred_d = dt.predict(x_train_d)
print("Train accuracy using desision trees with optimal parameters: ",accuracy_score(y_train_d,y_train_pred_d)) # compute the train accuracy


c_d = confusion_matrix(y_test_d,y_pred_d)     # We want to see the confusion matrix of DT  model
sns.heatmap(c_d, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = plt.axes() )
plt.title('Confusion Matrix using Decision Tree')
plt.show()

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]      # for plotting classification report of DT 
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-verginica']
print(classification_report(y_test_d,y_pred_d, target_names=classes))

plt.figure(figsize = (8,6))
plot_tree(dt, feature_names = features, class_names = classes, filled = True);
plt.show()

print("features importance: ",dt.feature_importances_)   # feature importance after applying DT model

# Now we want to plot the decision boundary for that we take only petal features 
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1,y.reshape(len(y),),test_size=0.25,random_state=0)  # use 75:25 train-test split

param1 = {'max_depth':[2,3,4,5,None]}
od1 = grid_search("d",param1,x_train1,y_train1,x_test1,y_test1)  # find optimal depth 

dt1= DecisionTreeClassifier(random_state = 0,max_depth = od1) # apply DT with optimal depth
dt1.fit(x_train1,y_train1.reshape(len(y_train1),))

plot_decision_regions(x1.astype(np.float),y.astype(np.integer).reshape(len(y),), clf=dt1, legend=2)  # plot 2-D decision boundary
plt.xlabel('petal length [cm]')
plt.ylabel('petal Width [cm]')
plt.title('Desision tree on Iris')
plt.show()

######################################### NAIVE BAYES ##############################################################

print("######################## NAIVE BAYES #######################################")
# We apply here Naive byes although we know Naive bayes assumes independent features but in our case from previous 
# results we find that petals are highly correlated. We want to know how robust is naive byes assumption is?
x_train2, x_test2, y_train2, y_test2 = train_test_split(x,y.reshape(len(y),),test_size=0.4,random_state=42) # use 60:40 train-split

gnb = GaussianNB()   # use sklearn GNB
gnb.fit(x_train2, y_train2.reshape(len(y_train2),))
y_pred_gnb = gnb.predict(x_test2)
acc_gnb = accuracy_score(y_test2,y_pred_gnb)
print("Test accuracy using Naive bayes model for all features: ",acc_gnb)  

# We want to see the results differ when we take only petals feature and also plot the decision boundary of GNB
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1,y.reshape(len(y),),test_size=.4,random_state=42)  

gnb1 = GaussianNB()
gnb1.fit(x_train1, y_train1.reshape(len(y_train1),))
y_pred_gnb1 = gnb1.predict(x_test1)
acc_gnb1 = accuracy_score(y_test1,y_pred_gnb1)
print("Test accuracy using Naive bayes model for only petal features: ",acc_gnb1)

plot_decision_regions(x1.astype(np.float),y.astype(np.integer).reshape(len(y),), clf=gnb1, legend=2) # plot 2-D decision boundary
plt.xlabel('petal length [cm]')
plt.xlabel('petal length [cm]')
plt.ylabel('petal Width [cm]')
plt.title('Naive byes on Iris')
plt.show()

# test accuracy of model using ony petal features is better than accuracy using all features which means that
# the model using all features may be overfitting the data.
# interestingly the accuracy using petal features is actually good rather than the fact that they are highly 
# correlated.

############################################## RANDOM FOREST #############################################################

print("######################## RANDOM FOREST #####################################")
# To avoid any kind of overfitting and to obtain a stable result unlike instability occur by a single tree.
# We want to use the random forest algorithm 
x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(x,y.reshape(len(y),),test_size=0.40,random_state=42)

param_r = {'max_depth':[2,3,4,5,6,8,10,None]}
m = grid_search("r",param_r,x_train_r,y_train_r,x_test_r,y_test_r)  # use implemented grid search in previous model

rf = RandomForestClassifier(max_depth = m,random_state=0,n_estimators=100) # apply RF for optimal parameters, 100 estimators should be enough
rf.fit(x_train_r, y_train_r.reshape(len(y_train_r),))
y_pred_rf = rf.predict(x_test_r)
acc_rf = accuracy_score(y_test_r,y_pred_rf)
print("Test accuracy using Random Forest for all features: ",acc_rf) # compute the test accuracy of the RF model

# We want to see the results differ when we take only petals feature and also plot the decision boundary of GNB 
x_train_r1, x_test_r1, y_train_r1, y_test_r1 = train_test_split(x1,y.reshape(len(y),),test_size=0.40,random_state=42)
rf1 = RandomForestClassifier(max_depth = m,random_state=0)
rf1.fit(x_train_r1, y_train_r1.reshape(len(y_train_r1),))
y_pred_rf1 = rf1.predict(x_test_r1)
acc_rf1 = accuracy_score(y_test_r1,y_pred_rf1)
print("Test accuracy using Random Forest for only petal features: ",acc_rf1)

plot_decision_regions(x1.astype(np.float),y.astype(np.integer).reshape(len(y),), clf=rf1, legend=2)  # plot 2-D decision boundary
plt.xlabel('petal length [cm]') 
plt.ylabel('petal Width [cm]')
plt.title('Random forest on Iris')
plt.show()

print("features importance: ",rf.feature_importances_)  # see the feature importance using RF model , is it differ from DT model

feature_imp = pd.Series(rf.feature_importances_,index=features).sort_values(ascending=False) # plot the feature importance for visualisation
plt.figure(figsize = (8,6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Contribution of features using Random forest")
plt.show()

############################################## KNN #############################################################

print("################################# KNN ######################################")
x_train_k, x_test_k, y_train_k, y_test_k = train_test_split(x,y.reshape(len(y),),test_size=0.40,random_state=42)

finding_k = list(range(1,50,2))   # find the optimal value for k 
scores = []

for k in finding_k:                                  # perform 10-fold cross validation
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, x_train_k, y_train_k, cv=10, scoring='accuracy')
    scores.append(score.mean())

error = [1 - x for x in scores]   # plot the (error vs different value of k) to see the optimal value of k
plt.figure(figsize=(12,8))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(finding_k, error)
plt.show()

best_k = finding_k[error.index(min(error))]
print("The optimal number of neighbors are: ",best_k)  # best value of k

KNN = KNeighborsClassifier(n_neighbors=best_k)  # apply knn for optimal value of k
KNN.fit(x_train_k, y_train_k)
y_pred_k = KNN.predict(x_test_k)

c_k = confusion_matrix(y_test_k, y_pred_k)  # Confusion matrix of KNN model
plt.figure(figsize = (6,5))
sns.heatmap(c_k, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = plt.axes() )
plt.title('Confusion Matrix using KNN')
plt.show()

acc_k = accuracy_score(y_test_k,y_pred_k)
print("Test accuracy using KNN for all features: ",acc_k)

# We want to see the results differ when we take only petals feature
x_train_k1, x_test_k1, y_train_k1, y_test_k1 = train_test_split(x1,y.reshape(len(y),),test_size=0.40,random_state=42)
KNN1 = KNeighborsClassifier(n_neighbors=best_k)
KNN1.fit(x_train_k1, y_train_k1)
y_pred_k1 = KNN1.predict(x_test_k1)

acc_k1 = accuracy_score(y_test_k,y_pred_k)
print("Test accuracy using KNN for only two petal features: ",acc_k1)

# there is no difference if we take only petals and all features.
############################################## SVM #############################################################

print("################################ SVM #######################################")
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x,y.reshape(len(y),),test_size=0.40,random_state=42)   # train test slpit 60:40

SVC = SVC(C=10,random_state = 42)   # implement SVC using initially regularization parameter C = 10 and random_state = 0 for same result.
SVC.fit(x_train_s,y_train_s)        # fit the model on all features 

#computes the score during the fit of an estimator on a parameter grid and chooses the parameters to maximize the cross-validation score.
finding_c = np.logspace(start=0,stop=10,num=100,base=2,dtype='float64')    # give values in log scale
par= {'C':finding_c}         

svc = GridSearchCV(SVC, param_grid =par, cv=10 ,scoring='accuracy')  # apply gridsearch to fit the estimator for best parameters
svc.fit(x_train_s, y_train_s)              # fit the gridsearch model
y_pred_s = svc.predict(x_test_s)

acc_s = accuracy_score(y_test_s,y_pred_s)
print("Test accuracy using SVC for all features: ",acc_s)     # calculate the accuracy on test data 

# Now we want to see if the results differ when we take only petal features 
x_train_s1, x_test_s1, y_train_s1, y_test_s1 = train_test_split(x1,y.reshape(len(y),),test_size=0.40,random_state=42) # x1 includes only petal features

SVC.fit(x_train_s1,y_train_s1)

svc1 = GridSearchCV(SVC, param_grid =par, cv=10 ,scoring='accuracy')
svc1.fit(x_train_s1, y_train_s1)
y_pred_s1 = svc1.predict(x_test_s1)

acc_s1 = accuracy_score(y_test_s1,y_pred_s1)       # calculate accuracy on test data using petal features 
print("Test accuracy using SVC for only two petal features: ",acc_s1)
plot_decision_regions(x1.astype(np.float),y.astype(np.integer).reshape(len(y),), clf=svc1, legend=2)    # plot desicion boundary of SVM 
plt.xlabel('petal length [cm]')
plt.ylabel('petal Width [cm]')
plt.title('Desision boundary on Iris using SVM')
plt.show()

# there is no difference if we take only petals and all features. Both gives good results.
#################################################### THE END #######################################################