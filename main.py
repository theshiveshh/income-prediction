import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('adult.csv')
print(df.head())
print(df.isnull().sum())


#print("df.dtypes",df.dtypes)
print("df.dtypes",df.dtypes)

df.drop('Native Country',axis=1,inplace=True) #no need of descrimination for the income
df.drop('Race',axis=1,inplace=True) #no need of descrimination for the income
print(" we are droping Native Country and Race as no need of country and colour descrimination for the income \n")
print("df.dtypes\n ",df.dtypes)

print('\n Workclass',df['Workclass'].nunique())
print('Material Status',df['Material Status'].nunique())
print('Relationship',df['Relationship'].nunique())
print('Occupation',df['Occupation'].nunique())
print('Education ',df['Education'].nunique())

from sklearn.preprocessing import LabelEncoder


lb = LabelEncoder()
df['Gender'] = lb.fit_transform(df['Gender'])
df['Workclass'] = lb.fit_transform(df['Workclass'])
df['Relationship'] = lb.fit_transform(df['Relationship'])
df['Material Status'] = lb.fit_transform(df['Material Status'])
df['Occupation'] = lb.fit_transform(df['Occupation'])
df['Education'] = lb.fit_transform(df['Education'])

#print(df['Workclass'].value_counts())

print(df.head())

x = df.drop('Income',axis=1)  # df.iloc[:,1:]
y = df['Income']
print(x.columns)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print("\n BUILDING THE MODELS\n ")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

def eval_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print("\nConfusion matrix\n",cm)
    print("\nClassification Report\n",classification_report(ytest,ypred))
    
    
def mscore(model):
    print('Training score',model.score(x_train,y_train))
    print('Testing score',model.score(x_test,y_test))

print("Logistic Regression Model\n ")

m1 = LogisticRegression(max_iter=1000)
m1.fit(x_train,y_train)

print("Accuracy \n")
mscore(m1)

ypred_m1 = m1.predict(x_test)
print("\nPredicted values : \n",ypred_m1)
eval_metrics(y_test,ypred_m1)



print("Validation :\n ")
accurcy1 = (5964+519)/(5964+206+1452+519)
print("Accurcy: ",accurcy1)
pre0 = 5964/(5964+1452)
pre1 = 519/(519+206)
rec0 = 5964/(5964+206)
rec1 = 519/(519+1452)
print("precision0",pre0)
print("precision1",pre1)
print("recall0",rec0)
print("recall1",rec1)
f1s0 = 2*pre0*rec0/(pre0 + rec0)
f1s1 = 2*pre1*rec1/(pre1 + rec1)
print('F1_Score0',f1s0)
print('F1_Score1',f1s1)


print("\nDecision Tree Model\n ")

m2 = DecisionTreeClassifier(criterion='gini',max_depth=6,min_samples_split=15)
m2.fit(x_train,y_train)

print("Acuuracy \n ")
mscore(m2)


ypred_m2 = m2.predict(x_test)
print("\nPredicted values : \n",ypred_m2)
eval_metrics(y_test,ypred_m2)

print("Validation :\n ")
accurcy2 = (5889+1042)/(5889+281+929+1042)
print("Accurcy: ",accurcy2)
pre02 = 5889/(5889+929)
pre12 = 1042/(1042+281)
rec02 = 5889/(5889+281)
rec12 = 1042/(1042+929)
print("precision0 ",pre02)
print("precision1 ",pre12)
print("recall0 ",rec02)
print("recall1 ",rec12)
f1s02 = 2*pre02*rec02/(pre02 + rec02)
f1s12 = 2*pre12*rec12/(pre12 + rec12)
print('F1_Score0 ',f1s02)
print('F1_Score1 ',f1s12)

print("\nRandom Forest Model\n ")

m3 = RandomForestClassifier(n_estimators=100,criterion='entropy',min_samples_split=20,max_depth=8)
m3.fit(x_train,y_train)

print("Acuuracy \n ")
mscore(m3)

ypred_m3 = m3.predict(x_test)
print("\nPredicted value: \n",ypred_m3)
eval_metrics(y_test,ypred_m3)

print("Validation :\n ")
accurcy3 = (5887+1076)/(5887+260+918+1076)
print("accurcy: ",accurcy3)
pre03 = 5887/(5887+918)
pre13 = 1076/(1076+260)
rec03 = 5887/(5887+260)
rec13 = 1076/(1076+918)
print("precision0",pre03)
print("precision1",pre13)
print("recall0",rec03)
print("recall1",rec13)
f1s03 = 2*pre03*rec03/(pre03 + rec03)
f1s13 = 2*pre13*rec13/(pre13 + rec13)
print('F1_Score0',f1s03)
print('F1_Score1',f1s13)

print("\nKNN Classifier Model\n ")

m4 = KNeighborsClassifier(n_neighbors=41)
m4.fit(x_train,y_train)

print("Acuuracy\n ")
mscore(m4)

ypred_m4 = m4.predict(x_test)
print("\nPredicted values: \n",ypred_m4)
eval_metrics(y_test,ypred_m4)

print("Validation :\n ")
accurcy4 = (6180+320)/(6180+30+1611+320)
print("accurcy: ",accurcy4)
pre04 = 6180/(6180+1611)
pre14 = 320/(320+30)
rec04 = 6180/(6180+30)
rec14 = 320/(320+1611)
print("precision0",pre04)
print("precision1",pre14)
print("recall0",rec04)
print("recall1",rec14)
f1s04 = 2*pre04*rec04/(pre04 + rec04)
f1s14 = 2*pre14*rec14/(pre14 + rec14)
print('F1_Score0',f1s04)
print('F1_Score1',f1s14)

print("\nSVM Model\n ")

m5 = SVC(kernel='linear',C=10)
m5.fit(x_train,y_train)

print("Acuuracy\n ")
mscore(m5)

ypred_m5 = m5.predict(x_test)
print("\nPredicted values : \n",ypred_m5)
eval_metrics(y_test,ypred_m5)

print("Validation :\n ")
accurcy4 = (6180+320)/(6180+30+1611+320)
print("accurcy: ",accurcy4)
pre05 = 6180/(6180+1611)
pre15 = 320/(320+30)
rec05 = 6180/(6180+30)
rec15 = 320/(320+1611)
print("precision0",pre05)
print("precision1",pre15)
print("recall0",rec05)
print("recall1",rec15)
f1s05 = 2*pre05*rec05/(pre05 + rec05)
f1s15 = 2*pre15*rec15/(pre15 + rec15)
print('F1_Score0',f1s05)
print('F1_Score1',f1s15)


print(" ' Random Forest Classifier' has the highest accuracy")









