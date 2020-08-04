# import pacakegs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import seaborn as sns
# print dasetset
frcheck = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Random Forest\\Fraud_check.csv")
frcheck.head()
colnames = list(frcheck.columns)

# change str data type into int data type
frcheck.loc[frcheck['Undergrad']=='YES','Undergrad']=1
frcheck.loc[frcheck['Undergrad']=='NO','Undergrad']=0

frcheck.loc[frcheck['Marital.Status']=='Single','Marital.Status']=1
frcheck.loc[frcheck['Marital.Status']=='Married','Marital.Status']=2
frcheck.loc[frcheck['Marital.Status']=='Divorced','Marital.Status']=3

frcheck.loc[frcheck['Urban']=='YES','Urban']=1
frcheck.loc[frcheck['Urban']=='NO','Urban']=0

frcheck.loc[frcheck['Taxable.Income']<=30000,'Taxable.Income']=1
frcheck.loc[frcheck['Taxable.Income']>30000,'Taxable.Income']=0

#plot bar graph
fc = frcheck['Taxable.Income'].value_counts()
plt.hist(frcheck['Taxable.Income'],edgecolor='k')
for i, v in enumerate(fc):
    plt.text(i, 
              v, 
             fc[i], 
              fontsize=18, 
              color="red")
plt.show()

ug = frcheck['Undergrad'].value_counts()
plt.hist(frcheck['Undergrad'],edgecolor='k')
for i, v in enumerate(ug):
    plt.text(i, 
              v, 
             ug[i], 
              fontsize=18, 
              color="red")
plt.show()

ur = frcheck['Urban'].value_counts()
plt.hist(frcheck['Urban'],edgecolor='k')    
for i, v in enumerate(ur):
    plt.text(i, 
              v, 
             ur[i], 
              fontsize=18, 
              color="red")
plt.show()

# check skewnee & kurtosis
# graph plot
frcheck['City.Population'].skew()
frcheck['City.Population'].kurt()
plt.hist(frcheck['City.Population'],edgecolor='k')
sns.distplot(frcheck['City.Population'],hist=False)
plt.boxplot(frcheck['City.Population'])

frcheck['Work.Experience'].skew()
frcheck['Work.Experience'].kurt()
plt.hist(frcheck['Work.Experience'],edgecolor='k')
sns.distplot(frcheck['Work.Experience'],hist=False)
plt.boxplot(frcheck['Work.Experience'])

sns.pairplot(frcheck)

# pie graph plot
frcheck.Undergrad.value_counts().plot(kind="pie")
frcheck.Urban.value_counts().plot(kind="pie")

#split train & test dataset
y = frcheck['Taxable.Income']
x = frcheck.drop(['Taxable.Income'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#convert datatype into int.
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


































