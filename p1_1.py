from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def evaluate( trainX, trainY, testX, testY): 
    PrtrainY=trainX
    PrtestY=testX
    rmse_tr=np.sqrt(mean_squared_error(trainY, PrtrainY))
    rmse_te=np.sqrt(mean_squared_error(testY, PrtestY))
    mae_tr=mean_absolute_error(trainY, PrtrainY)
    mae_te=mean_absolute_error(testY, PrtestY)
    print("RMSE train:", rmse_tr)
    print("RMSE test:", rmse_te)
    print("MAE train:", mae_tr)
    print("MAE test:", mae_te)
    return rmse_tr, rmse_te, mae_tr, mae_te

data=pd.read_csv('pdData.csv',index_col=0) #coloumn 0 is instance number not a feature!
X=data.iloc[:,:-1]
y=data["y"]

#spilit data :
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5, random_state=1) 

#insert coloumn 1..1
#s=trainX.drop(trainX.columns[0],axis=1)
#X = s.assign(a=1)[['a'] + s.columns.tolist()]
#z=testX.drop(testX.columns[0],axis=1)
#tsX= z.assign(a=1)[['a'] + z.columns.tolist()]
tsX=testX

#calculate w:
#wLS=((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(trainY)
wLS=np.linalg.lstsq(trainX,trainY,rcond=None)[0]
print("Coefficients of ten first features according to my implementation:",wLS[0:10])
print("")

#bulid model :
reg=LinearRegression(fit_intercept=False)
reg.fit(trainX,trainY)
print("Euclidean norm of distance between my coef and model's coef:",np.linalg.norm(wLS-reg.coef_)) #without w0
print("")
for i in range(99):
    print("my_coef(",i,")-reg_coef(",i,")=",abs(wLS[i]-reg.coef_[i]))
print("")

#evaluate:
PrtrainY=trainX.dot(wLS.T)
PrtestY=tsX.dot(wLS.T)
print("Evaluation based on my coef")
print("")
evaluate(PrtrainY,trainY,PrtestY,testY)
print("")

PrtrainY=trainX.dot(reg.coef_.T)
PrtestY=tsX.dot(reg.coef_.T)
print("Evaluation based on model's coef")
print("")
evaluate(PrtrainY,trainY,PrtestY,testY)

    





