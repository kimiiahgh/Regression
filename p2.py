#الف و ب و ج
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def evaluate( PrtrainY, trainY, PrtestY, testY): 
    rmse_tr=np.sqrt(mean_squared_error(trainY, PrtrainY))
    rmse_te=np.sqrt(mean_squared_error(testY, PrtestY))
    mae_tr=mean_absolute_error(trainY, PrtrainY)
    mae_te=mean_absolute_error(testY, PrtestY)
    print("RMSE train:", rmse_tr)
    print("RMSE test:", rmse_te)
    print("MAE train:", mae_tr)
    print("MAE test:", mae_te)
    return [rmse_tr, rmse_te, mae_tr, mae_te]

def Polynominal(trainX,trainY,testX,testY,M):
    model = np.poly1d(np.polyfit(trainX, trainY, M))
    y_pred_tr=[]
    y_pred_te=[]
    for i in trainX:
        y_pred_tr.append(model(i))
    for j in testX:
        y_pred_te.append(model(j))        
    y_pred_tr=pd.DataFrame(y_pred_tr) 
    y_pred_te=pd.DataFrame(y_pred_te) 
    print("__________________")
    print("Evaluation of Polynominal Regression with degree=",M,":")
    print("")
    evaluate(y_pred_tr,trainY,y_pred_te,testY)
def PHiTPHi(trainX,M):
    phi=[]
    for i in trainX:
        x=[1]
        for j in range(1,M+1):
           x.append(pow(i,j)) 
        phi.append(x)
    phi=np.array(phi)
    matrix=np.dot(phi.transpose(),phi)
    return matrix 

#########
data1=pd.read_csv('data1.csv') 
X1=data1["x"]
y1=data1["t"]
data2=pd.read_csv('data2.csv') 
X2=data2["x"]
y2=data2["t"]
data3=pd.read_csv('data3.csv') 
X3=data3["x"]
y3=data3["t"]
data4=pd.read_csv('data4.csv') 
X4=data4["x"]
y4=data4["t"]
data5=pd.read_csv('data5.csv') 
X5=data5["x"]
y5=data5["t"]
##########

#calculate rmse and me for different M using data1 and data2 and condition number  :
phi1=[]
phi2=[]
for i in range(1,10):

    print("__________________")
    print("using data1 as train data and data3 as test data ")
    Polynominal(X1,y1,X3,y3,i)
    print("Condition number of matrix is:",np.linalg.cond(PHiTPHi(X1,i)))
    phi1.append(math.log(np.linalg.cond(PHiTPHi(X1,i))))
    print("__________________")

    print("using data2 as train data and data3 as test data ")
    Polynominal(X2,y2,X3,y3,i)
    print("Condition number of matrix is:",np.linalg.cond(PHiTPHi(X2,i)))
    phi2.append(math.log(np.linalg.cond(PHiTPHi(X2,i))))
    print("")
M=[i for i in range(1,10)]
    
#plot M and cond(A) 
plt.figure()
plt.scatter(phi1, M)
plt.xlabel(" logarithm of condition number of matrix using data 1")
plt.ylabel("M")
plt.show(block=False)

plt.figure()
plt.scatter(phi2, M)
plt.xlabel(" logarithm of condition number of matrix using data 2")
plt.ylabel("M")
plt.show(block=False)

#Regularization :
#د
phiTr=[]
for i in X2:
    x=[1]
    for j in range(1,10):
       x.append(pow(i,j)) 
    phiTr.append(x)
phiTr=pd.DataFrame(phiTr)
print(phiTr.shape)

phiTs=[]
for i in X3:
    x=[1]
    for j in range(1,10):
       x.append(pow(i,j)) 
    phiTs.append(x)
phiTs=pd.DataFrame(phiTs)    

#norm 2:
print("__________________")
print("norm 2 using data2 as train and data3 as test")
a=np.arange(1.e-7,1.e-5,1.e-7)
landa=[(10.e+6)*i for i in a]
wLanda=[]
Aw_t=[]
E1=[]
E2=[]

for i in a:
    
    reg = linear_model.Ridge(alpha=i)
    reg.fit(phiTr,y2)
    w=reg.coef_
    wLanda.append(math.log(pow(np.linalg.norm(w),2)))
    y_y=phiTr.dot(w)-y2
    Aw_t.append(pow(np.linalg.norm(y_y),2))
    y_pred_tr=reg.predict(phiTr)
    y_pred_te=reg.predict(phiTs)
    print("Evaluation of Polynominal Regression with degree=9 and penalty=",i,":")
    print("")
    e=evaluate(y_pred_tr,y2,y_pred_te,y3)
    E1.append(e[0]* 10.e+2)
    E2.append(e[1]*10.e+2)
bestLanda1=a[E2.index(max(E2))]
bestLanda2=a[E2.index(min(E2))]
print("best landa is:",bestLanda2)

    
    
    
#figure  w, landa, Aw-t 
plt.figure()
plt.scatter(E1,landa)
plt.xlabel("RMSE train using norm2")
plt.ylabel("Landa")
plt.show(block=False)


plt.figure()
plt.scatter(E2,landa)
plt.xlabel("RMSE test using norm2")
plt.ylabel("Landa")
plt.show(block=False)    
    
    
plt.figure()
plt.scatter(wLanda,landa)
plt.xlabel("||w||2^2 using norm2")
plt.ylabel("Landa")
plt.show(block=False)

plt.figure()
plt.scatter(Aw_t,landa)
plt.xlabel("||Aw-t||2^2 using norm2")
plt.ylabel("Landa")
plt.show(block=False)

#norm 1:
print("__________________")
print("norm 1 using data2 as train and data3 as test")
wLanda=[]
Aw_t=[]
E1=[]
E2=[]
for i in a:
    reg = linear_model.Lasso(alpha=i)
    reg.fit(phiTr,y2)
    w=reg.coef_
    wLanda.append(math.log(pow(np.linalg.norm(w),2)))
    y_y=phiTr.dot(w)-y2
    Aw_t.append(pow(np.linalg.norm(y_y),2))
    y_pred_tr=reg.predict(phiTr)
    y_pred_te=reg.predict(phiTs)
    print("Evaluation of Polynominal Regression with degree=9 and penalty=",i,":")
    print("")
    e=evaluate(y_pred_tr,y2,y_pred_te,y3)
    E1.append(e[0]* 10.e+2)
    E2.append(e[1]*10.e+2)

bestLanda1=a[E2.index(max(E2))]
bestLanda2=a[E2.index(min(E2))]
print("best landa is:",bestLanda2)
    
#figure  w, landa, Aw-t 
    
plt.figure()
plt.scatter(E1,landa)
plt.xlabel("RMSE train using norm1")
plt.ylabel("Landa")
plt.show(block=False)


plt.figure()
plt.scatter(E2,landa)
plt.xlabel("RMSE test using norm1")
plt.ylabel("Landa")
plt.show(block=False)    
        
    
plt.figure()
plt.scatter(wLanda,landa)
plt.xlabel("||w||2^2 using norm1")
plt.ylabel("Landa")
plt.show(block=False)

plt.figure()
plt.scatter(Aw_t,landa)
plt.xlabel("||Aw-t||2^2 using norm1")
plt.ylabel("Landa")
plt.show(block=False)

#RidgeCV using data2:
from sklearn.linear_model import RidgeCV
reg = RidgeCV(alphas=a).fit(phiTr, y2)
print("___________________")
alpha=reg.alpha_
print("using data 2 as train")
print("the best landa based on RidgeCV is :",alpha)


#bulid model with M=9 using data5 :
print("_________________")
print("using data5 as train data and data3 as test data ")
Polynominal(X5,y5,X3,y3,9)
print("Condition number of matrix is:",np.linalg.cond(PHiTPHi(X5,9)))
print("") #moqayese tuye tahlil

#M=8 without regularization : 
phiTr=[]
for i in X4:
    x=[1]
    for j in range(1,10):
       x.append(pow(i,j)) 
    phiTr.append(x)
phiTr=pd.DataFrame(phiTr)
phiTs=[]
for i in X3:
    x=[1]
    for j in range(1,10):
       x.append(pow(i,j)) 
    phiTs.append(x)
phiTs=pd.DataFrame(phiTs) 
#norm 2
reg = linear_model.Ridge()
reg.fit(phiTr,y4)
y_pred_tr=reg.predict(phiTr)
y_pred_te=reg.predict(phiTs)
print("__________________")
print("norm 2 using data4 as train and data3 as test")
print("Evaluation of Polynominal Regression with degree=8")
print("")
evaluate(y_pred_tr,y4,y_pred_te,y3)
#norm 1
reg = linear_model.Lasso()
reg.fit(phiTr,y4)
y_pred_tr=reg.predict(phiTr)
y_pred_te=reg.predict(phiTs)
print("__________________")
print("norm 1 using data4 as train and data3 as test")
print("Evaluation of Polynominal Regression with degree=8")
print("")
evaluate(y_pred_tr,y4,y_pred_te,y3)

#ز
#Condition number phi with regularization
phi=PHiTPHi(X2,9) #M=9
a=np.arange(1.e-7,1.e-5,1.e-7)
I=np.identity(10)
cnd=[]
print("____________________________")
print("using data2 as train and M=9 with regularization")
for i in a:
    matrix=phi+(I*i)
    print("Condition number is:",np.linalg.cond(matrix))
    cnd.append(math.log(np.linalg.cond(matrix)))
landa=[(10.e+6)*i for i in a]
plt.figure()
plt.scatter(cnd, landa)
plt.xlabel(" logarithm of condition number of matrix using data 2 ")
plt.ylabel("Landa for regularization")
plt.show(block=False)    
    
    







