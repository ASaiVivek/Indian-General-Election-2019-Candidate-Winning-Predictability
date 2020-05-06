#Sai Vivek Amirishetty- https://github.com/vivekboss99/Indian-General-Election-2019-Candidate-Winning-Predictability.git -for dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

elections=pd.read_csv("LS.csv")
#print(elections.head())

#elections['EDUCATION'].value_counts().plot.bar()
elections.dropna(inplace=True)


def changeValues(x):
    try:
        temp=(x.split('Rs')[1].split('\n')[0].strip())
        temp2=temp.replace(',','')
        return temp2
    except:
        x=0
        return x
elections['ASSETS']=elections['ASSETS'].apply(changeValues)
elections['LIABILITIES']=elections['LIABILITIES'].apply(changeValues)
#print(elections['LIABILITIES'].head())
#print(elections['ASSETS'].head())

elections['ASSETS']=elections['ASSETS'].astype('int64')
elections['LIABILITIES']=elections['LIABILITIES'].astype('int64')

l=elections.select_dtypes('object').columns #WHYYY
lb=LabelEncoder()
for i in l:
    elections[i]=lb.fit_transform(elections[i])

elections.drop(['NAME'],axis=1,inplace=True)

X=elections.drop('WINNER',axis=1)
Y=elections['WINNER']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
predict=model.predict(X_test)
score=(accuracy_score(predict,Y_test))
#score=cross_val_score(estimator=model,X=X_train,y=Y_train,cv=10)
print(score)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
rfc=RandomForestClassifier(n_estimators=150)
rfc.fit(X_train,Y_train)
pred_rfc=rfc.predict(X_test)
rfc_eval=cross_val_score(estimator=rfc,X=X_train,y=Y_train,cv=10)
#score1=accuracy_score(pred_rfc,Y_test)
print(rfc_eval.mean())
