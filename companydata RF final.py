# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import seaborn as sns
# print dataset
comydata = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Random Forest\\Company_Data.csv")
comydata.head()
colnames = list(comydata.columns)
# change str data type into unique value int data type
comydata.loc[comydata['ShelveLoc']=='Good','ShelveLoc']=0
comydata.loc[comydata['ShelveLoc']=='Bad','ShelveLoc']=1
comydata.loc[comydata['ShelveLoc']=='Medium','ShelveLoc']=2

comydata.loc[comydata['Urban']=='Yes','Urban']=1
comydata.loc[comydata['Urban']=='No','Urban']=0

comydata.loc[comydata['US']=='Yes','US']=1
comydata.loc[comydata['US']=='No','US']=0
# value count
comydata['Sales'].value_counts()

#plot bar graph
co = comydata['ShelveLoc'].value_counts()
plt.hist(comydata['ShelveLoc'],edgecolor='k')
plt.grid()
plt.show()

u = comydata['US'].value_counts()
plt.hist(comydata['US'],edgecolor='k')
plt.grid()
plt.show()

ur = comydata['Urban'].value_counts()
plt.hist(comydata['Urban'],edgecolor='k')    
plt.grid()
plt.show()

# check skewnee, kurtosis & plot graph
# sales
comydata['Sales'].skew()
comydata['Sales'].kurt()
plt.hist(comydata['Sales'],edgecolor='k')
sns.distplot(comydata['Sales'],hist=False)
plt.boxplot(comydata['Sales'])
# compprice
comydata['CompPrice'].skew()
comydata['CompPrice'].kurt()
plt.hist(comydata['CompPrice'],edgecolor='k')
sns.distplot(comydata['CompPrice'],hist=False)
plt.boxplot(comydata['CompPrice'])
# population
comydata['Population'].skew()
comydata['Population'].kurt()
plt.hist(comydata['Population'],edgecolor='k')
sns.distplot(comydata['Population'],hist=False)
plt.boxplot(comydata['Population'])
# pair plot
sns.pairplot(comydata)

# pie graph plot
comydata.ShelveLoc.value_counts().plot(kind="pie")
comydata.Urban.value_counts().plot(kind="pie")

#split train & test dataset
y = comydata['Sales']
x = comydata.drop(['Sales'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# convert datatype into int.
X_train = X_train.astype("int")
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

#implement random forest classifier with different hyperparameter.
# n_estimators=50,criterion='entropy',max_features='sqrt'
clf=RandomForestClassifier(n_estimators=50,criterion='entropy',max_features='sqrt')
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
score=metrics.balanced_accuracy_score(Y_test, Y_pred)
print(score)
# n_estimators=100,criterion='entropy',max_features='log2'
clf1=RandomForestClassifier(n_estimators=100,criterion='entropy',max_features='log2')
clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)
score1=metrics.balanced_accuracy_score(Y_test, Y_pred)
print(score1)
# n_estimators=100,max_features='log2',max_depth=100
clf2=RandomForestClassifier(n_estimators=100,max_features='log2',max_depth=100)
clf2.fit(X_train,Y_train)
Y_pred=clf2.predict(X_test)
score2=metrics.balanced_accuracy_score(Y_test, Y_pred)
print(score2)

# Adaboost classifier
# n_estimators=100, random_state=0
clf5 = AdaBoostClassifier(n_estimators=100, random_state=0)
clf5.fit(X_train,Y_train)
Y_pred=clf5.predict(X_test)
score2=metrics.balanced_accuracy_score(Y_test, Y_pred)
print(score2)























