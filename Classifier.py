## Code for classification models and preprocessing of image stream dataset


#main libraries
#filter some warnings on numpy old functions- do not cause any problems
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time


## import random library
import random
## import the SKLearn library for SVM and logistic regression
from sklearn import svm
#imported sklearn libraries for validation and metrics. All metrics explained in comments.
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

## SVM code adapted from Data Mining SVM Lab and is referenced in report

## X is the features vector (3 features)
X = np.array(pd.read_csv('Data Files/WLA.csv'))
## y is the labels/output - single integer value (0,1)
y = np.array(pd.read_csv("Data Files/Labels.csv"))


#column labels
cols = ["Width", "Length", 'Size']
#show correlation of features
Xtemp = pd.DataFrame(X)
Xtemp.columns = cols
corrmat = Xtemp.corr().abs()
plt.figure(figsize=(6,4))
sns.heatmap(corrmat,cmap=plt.cm.Reds)
plt.title("Correlation of Features")
plt.show()
#Extreme correlation between size and width & length respectively due to nature of creation
#Must remove and use first 2
## add names for the labels
target_names = ['Motorcycle', 'Car']
X = X[:,0:2]
#deletion of first 16 rows
X = np.delete(X,range(0,15),0)
y = np.delete(y,range(0,15),0)
#length of samples
n_sample = len(X)

#set seed for repeatability
np.random.seed(0)
## get the index for X and y
index = [i for i in range(n_sample)]
## shuffle the index
random.shuffle(index)
##split the index into 70% for training and 30% for testing
train_index = index[:int(0.7*n_sample)]
test_index = index[int(0.7*n_sample):]
##Using the index to split the data
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index].astype(float), y[test_index].astype(float)



# use 3 kernels (linear, RBF and polynomial from the build in SVM routine of SKLearn)
for kernel in ('linear', 'poly', 'rbf'):
    print(" ")
    print(kernel+" SVM")
    #record time to train each model
    start = time.time()
    #paramaters for rbf only
    #gamma choice discussed in report
    #C is relatively small to allow more relaxed decision boundary - not as good at classifying training as a result but better on test data
    clf = svm.SVC(kernel=kernel, gamma=0.0008, C=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end = time.time()
    print(kernel + " SVM time: " + str(end - start))
      
    ## plot the results
    plt.figure()
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Pastel1,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')
    plt.legend(labels=["Training", "Test"], loc='upper left')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Pastel1)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-1, 0, 1])

    plt.title(kernel)
    plt.show()
    
    #plot confusion matrix
    plt.figure()
    confMatrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(confMatrix, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion matrix for " + kernel + " SVM")
    plt.show()
    
    #close to or definitely 100% classification 
    #k-fold cross validation on all metrics!!!!
    cv = KFold(n_splits=10, shuffle=True)
    #accuracy computes (TP+TN) / (number of points), gives number of correct predictions from total
    acc = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    #precision computes TP/(TP+FP),gives true predictions from predicted classes
    prec = cross_val_score(clf, X, y, scoring='precision', cv=cv, n_jobs=-1)
    #recall computes TP/(TP+FN), gives true predictions from actual classes
    rec = cross_val_score(clf, X, y, scoring='recall', cv=cv, n_jobs=-1)
    #f1 combines precision and recall because we want a balance of these two metrics, 2*(Recall * Precision) / (Recall + Precision)
    f1 = cross_val_score(clf, X, y, scoring='f1', cv=cv, n_jobs=-1)
    
    #print scores
    print(kernel + " Accuracy: " + str(round(acc.mean()*100.0,2)))
    print(kernel + " Precision: " + str(round(prec.mean()*100.0,2)))
    print(kernel +" Recall: " + str(round(rec.mean()*100.0,2)))
    print(kernel + "F1: " + str(round(f1.mean()*100.0,2)))



##logistic regression from sklearn
from sklearn.linear_model import LogisticRegression
#used for roc curves (not actually used in report)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#time trained model
start = time.time()
logModel = LogisticRegression(random_state=1)
logModel.fit(X_train, y_train)
yPred = logModel.predict(X_test)
end = time.time()
print("Logistic Regression time: " + str(end - start))
#confusion matrix
confMatrix = confusion_matrix(y_test, yPred)
ax = sns.heatmap(confMatrix, annot=True, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion matrix for Logistic Regression")
plt.show()

#help from https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/ for plot
#get intercept and coefficients
b = logModel.intercept_[0]
B1, B2 = logModel.coef_.T
#calculate gradient and intercept of boundary
c = -b/B2
m = -B1/B2
#limits of graph
ymin = 10
ymax = 110
xmin = 0
xmax = 60
#colormap 
colormap = np.array(['g', 'orange', 'b'])
#getxs and ys for line using limits
xs = np.array([xmin,xmax])
ys = m*xs + c
#plot line and fill colours
plt.plot(xs, ys, 'k', lw=1, ls='--')
plt.fill_between(xs, ys, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xs, ys, ymax, color='tab:orange', alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=colormap[y[:,0]], s=8, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10, edgecolor='k')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel("Length")
plt.xlabel("Width")

#help from https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/ 
# to plot roc curves
#not referended as not actually used - classifiers are too accurate so ROC curve not needed in analysis
#visualisation is meaniningless
# lr_probs = logModel.predict_proba(X_test)[:,1]
# auc = roc_auc_score(y_test,lr_probs)
# lrSens, lrSpec, _ = roc_curve(y_test,lr_probs)
# plt.figure()
# plt.plot(lrSpec, lrSens, label='Logistic Regression')
# plt.xlabel("1-Specificity")
# plt.ylabel("Sensitivity")
# plt.legend()
# plt.show()



#close to or definitely 100% classification 
#k-fold cross validation on all metrics!!!!
logModel = LogisticRegression(random_state=1)
cv = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(logModel, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
prec = cross_val_score(logModel, X, y, scoring='precision', cv=cv, n_jobs=-1)
rec = cross_val_score(logModel, X, y, scoring='recall', cv=cv, n_jobs=-1)
f1 = cross_val_score(logModel, X, y, scoring='f1', cv=cv, n_jobs=-1)
#Accuracy
print(" ")
print("Logistic Regression Accuracy: " + str(round(acc.mean()*100.0,2)))
print("Logistic Regression Precision: " + str(round(prec.mean()*100.0,2)))
print("Logistic Regression Recall: " + str(round(rec.mean()*100.0,2)))
print("Logistic Regression F1: " + str(round(f1.mean()*100.0,2)))




