import pandas as pd

from sklearn.neighbors import KNeighborsClassifier 

def switch(data):
    

    if data==[2]:
        return 'Benign'
    
    if data==[4]:
        return 'Malignant'

GetRawData=pd.read_csv('./breast-cancer-wisconsin.csv',usecols=[1,2,3,4,5,6,7,8,9,10])

Features=GetRawData.drop(columns='Class')
lable=GetRawData['Class']
BulidClassifier=KNeighborsClassifier(n_neighbors=1)
BulidClassifier.fit(X=Features,y=lable)

print(switch(BulidClassifier.predict([[1,1,1,1,5,1,3,1,1]])))


