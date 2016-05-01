import numpy as np
import scipy as sci
from sklearn.preprocessing import scale
from sklearn import linear_model, neighbors, preprocessing, cross_validation, svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from matplotlib import pyplot as plt

tempD = np.loadtxt("exam.dat.txt",dtype=np.str_,delimiter=" ")
row, col = tempD.shape
for i in range(row):
    for j in range(1,col):
        tempD[i][j] = tempD[i][j].split(":")[1]

y = np.array(tempD[0:,0],dtype=float)
X = np.array((tempD[0:,1:]),dtype=float)
X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)

##### Data Split #####
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

##### Logistic Model #####
logit_score = []
c = np.arange(0.5,50,0.5)
for i in c:
    logit_clf = linear_model.LogisticRegression(C=i)
    scores = cross_validation.cross_val_score(logit_clf, X_train, y_train, cv=5)
    val = np.mean(scores)
    logit_score.append(val)
index = logit_score.index(max(logit_score))
i = c[index]
logistic = linear_model.LogisticRegression(C=i)
logistic.fit(X_train,y_train)
yhat = logistic.predict(X_test)
terror = np.sum(np.abs(yhat - y_test))*1.0/len(X_test)
success= 1-terror
print success, i
print logit_score

##### Linear SVM #####
lsvm_score = []
c = np.arange(0.1,5.1,0.1)
for i in c:
    clf = svm.SVC(kernel='linear', C=i)
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    val = np.mean(scores)
    lsvm_score.append(val)
index = lsvm_score.index(max(lsvm_score))
i = c[index]
lsvm = svm.SVC(kernel='linear', C=i)
lsvm.fit(X,y)
yhat = lsvm.predict(X_test)
terror = np.sum(np.abs(yhat - y_test))*1.0/len(X_test)
success = 1 - terror
print success, i

##### rbf SVM #####
rsvm_tscore = []
c = np.arange(0.1,5.1,0.1)
G = np.arange(10.)*0.2+0.1

for i in c:
    temp_score = []
    for j in G:
        clf = svm.SVC(kernel='rbf', C = i, gamma = j)
        scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
        val = np.mean(scores)
        temp_score.append(val)
    rsvm_tscore.append(temp_score)
rsvm_score = np.array(rsvm_tscore)


s_max = np.amax(rsvm_score)
pos = np.where(rsvm_score == s_max)
i = c[pos[0][0]]
j = G[pos[1][0]]
rsvm = svm.SVC(kernel='rbf', C = i, gamma = j)
rsvm.fit(X,y)
yhat = rsvm.predict(X_test)
terror = np.sum(np.abs(yhat - y_test))*1.0/len(X_test)
success = 1 - terror

### ROC ###
rsvm = svm.SVC(kernel='rbf', C = i, degree = j)
y_test_score = rsvm.fit(X,y).decision_function(X_test)
false_posi_rate, true_posi_rate, thresholds = roc_curve(y_test, y_test_score)
rocauc = auc(false_posi_rate, true_posi_rate)

plt.figure()
plt.plot(false_posi_rate, true_posi_rate, 'b',label='AUC = %0.5f'% rocauc,
        linewidth = 2)
plt.title('ROC of rbf SVM (C = '+str(1)+', gamma = '+str(j)+' )',size =20)
plt.plot([0,1],[0,1],'r--',linewidth = 2)
plt.legend(loc='lower right')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate',size=20)
plt.xlabel('False Positive Rate',size=20)
plt.legend(numpoints=15,prop={'size':20},loc="lower right",frameon=True)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.grid(True)
plt.savefig("rsvm_roc.png")

### PR ###
precision, recall, thesh = precision_recall_curve(y_test, y_test_score)

plt.figure()
plt.title('PR of rbf SVM (C = '+str(1)+', gamma = '+str(j)+' )', size=20)
plt.plot(recall, precision, label='Precision-Recall curve',linewidth = 2)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('precision',size=20)
plt.xlabel('recall',size=20)
plt.legend(numpoints=15,prop={'size':20},loc="lower right",frameon=True)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.grid(True)
plt.savefig("rsvm_pr.png")

print success, i, j
print confusion_matrix(y_test, yhat)
print classification_report(y_test, yhat, digits=5)


##### poly SVM #####
psvm_tscore = []
c = np.arange(0.1,5.1,0.1)
D = np.arange(4)

for i in c:
    temp_score = []
    for j in D:
        clf = svm.SVC(kernel='poly', C = i, degree = j)
        scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
        val = np.mean(scores)
        temp_score.append(val)
    psvm_tscore.append(temp_score)
psvm_score = np.array(psvm_tscore)


s_max = np.amax(psvm_score)
pos = np.where(psvm_score == s_max)
i = c[pos[0][0]]
j = D[pos[1][0]]

psvm = svm.SVC(kernel='poly', C = i, degree = j)
psvm.fit(X,y)
yhat = psvm.predict(X_test)
terror = np.sum(np.abs(yhat - y_test))*1.0/len(X_test)
success = 1 - terror

### ROC ###
psvm = svm.SVC(kernel='poly', C = i, degree = j)
y_test_score = psvm.fit(X,y).decision_function(X_test)
false_posi_rate, true_posi_rate, thresholds = roc_curve(y_test, y_test_score)
rocauc = auc(false_posi_rate, true_posi_rate)

plt.figure()
plt.plot(false_posi_rate, true_posi_rate, 'b',label='AUC = %0.5f'% rocauc,
        linewidth = 2)
plt.title('ROC of polynomial SVM (C = '+str(1)+', degree = '+str(j)+' )',size =20)
plt.plot([0,1],[0,1],'r--',linewidth = 2)
plt.legend(loc='lower right')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate',size=20)
plt.xlabel('False Positive Rate',size=20)
plt.legend(numpoints=15,prop={'size':20},loc="lower right",frameon=True)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.grid(True)
plt.savefig("psvm_roc.png")

### PR ###
precision, recall, thesh = precision_recall_curve(y_test, y_test_score)

plt.figure()
plt.title('PR of polynomial SVM (C = '+str(1)+', degree = '+str(j)+' )', size=20)
plt.plot(recall, precision, label='Precision-Recall curve',linewidth = 2)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('precision',size=20)
plt.xlabel('recall',size=20)
plt.legend(numpoints=15,prop={'size':20},loc="lower right",frameon=True)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.grid(True)
plt.savefig("psvm_pr.png")


print success, i, j
print confusion_matrix(y_test,yhat)
print classification_report(y_test,yhat,digits=5)