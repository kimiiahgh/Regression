from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from pyearth import Earth
import statsmodels.api as sm
import pandas as pd
import numpy as np
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
########
rmse_tr=[]
rmse_te=[]
mae_tr=[]
mae_te=[]
########
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

def Polynominal(trainX,trainY,testX,testY):
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(trainX)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, trainY)
    y_pred_tr=pol_reg.predict(poly_reg.fit_transform(trainX))
    y_pred_te=pol_reg.predict(poly_reg.fit_transform(testX))
    y_pred_tr=pd.DataFrame(y_pred_tr) 
    y_pred_te=pd.DataFrame(y_pred_te) 
    print("")
    print("Evaluation of Polynominal Regression with degree=2 :")
    print("")
    e=evaluate(y_pred_tr,trainY,y_pred_te,testY)
    return e
    
def Ridge_(trainX,trainY,testX,testY):
    model = Ridge(alpha=1.0)
    model.fit(trainX, trainY)
    y_pred_tr=model.predict(trainX)
    y_pred_te=model.predict(testX)
    print("")
    print("Evaluation of Ridge Regression :")
    print("")
    e=evaluate(y_pred_tr,trainY,y_pred_te,testY)
    return e

def Lasso(trainX,trainY,testX,testY):
    model = linear_model.Lasso(alpha=0.1)
    model.fit(trainX,trainY)
    y_pred_tr=model.predict(trainX)
    y_pred_te=model.predict(testX)
    print("")
    print("Evaluation of Lasso Regression :")
    print("")
    e=evaluate(y_pred_tr,trainY,y_pred_te,testY)
    return e

def Elastic_Net(trainX,trainY,testX,testY):
    model = ElasticNet(random_state=0)
    model.fit(trainX, trainY)
    y_pred_tr=model.predict(trainX)
    y_pred_te=model.predict(testX)
    print("")
    print("Evaluation of Elastic Net Regression :")
    print("")
    e=evaluate(y_pred_tr,trainY,y_pred_te,testY)
    return e

def forward_regression(X, y,threshold_in,verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            #if verbose:
               # print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break
    return included

def backward_regression(X, y,threshold_out,verbose=False):
    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            #if verbose:
                #print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included    
def Spline_Regression(trainX,trainY,testX,testY):
    model = Earth()
    model.fit(trainX,trainY)
    y_pred_tr=model.predict(trainX)
    y_pred_te=model.predict(testX)
    print("")
    print("Evaluation of Spline Regression :")
    print("")
    e=evaluate(y_pred_tr,trainY,y_pred_te,testY)
    return e

#########
data=pd.read_csv('pdData.csv') #coloumn 0 is instance number not a feature!
data.drop(data.columns[0],axis=1)
X=data.drop("y",axis=1)
y=data["y"]
    
#spilit data :
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5, random_state=1) 
"""X=trainX.drop(trainX.columns[0],axis=1)
teX=testX.drop(testX.columns[0],axis=1)"""

#build model for polynominal regression:
e=Polynominal(trainX,trainY,testX ,testY)
rmse_tr.append(e[0])
rmse_te.append(e[1])
mae_tr.append(e[2])
mae_te.append(e[3])

#bulid model for Ridge regression:
e=Ridge_(trainX,trainY,testX,testY)
rmse_tr.append(e[0])
rmse_te.append(e[1])
mae_tr.append(e[2])
mae_te.append(e[3])

#build model for Lasso regression
e=Lasso(trainX,trainY,testX,testY)
rmse_tr.append(e[0])
rmse_te.append(e[1])
mae_tr.append(e[2])
mae_te.append(e[3])

#bulid model for Elastic Net regression:
e=Elastic_Net(trainX,trainY,testX,testY)
rmse_tr.append(e[0])
rmse_te.append(e[1])
mae_tr.append(e[2])
mae_te.append(e[3])

#build model for backward regression:
print("")
print("Evaluation of backward regression is:")
r=np.arange(0.1,1,0.1)
for i in r:
    print("")
    print("the threshold is :",i)
    step_data=backward_regression(trainX,trainY,i,verbose=False)
    x_step=trainX[step_data]
    reg=LinearRegression()
    print("the number of selected features are:",x_step.shape[1])
    reg.fit(trainX,trainY)
    reg.fit(x_step,trainY)
    y_pred_tr=reg.predict(x_step)
    y_pred_te=reg.predict(testX[step_data])
    e=evaluate(y_pred_tr,trainY,y_pred_te,testY)
    if i==0.5:
        rmse_tr.append(e[0])
        rmse_te.append(e[1])
        mae_tr.append(e[2])
        mae_te.append(e[3])
    
#bulid model for forward regression:
print("")
print("Evaluation of forward regression is:")
r=np.arange(0.1,1,0.1)
for i in r:
    print("")
    print("the threshold is :",i)
    step_data=forward_regression(trainX,trainY,i,verbose=False)
    x_step=trainX[step_data]
    reg=LinearRegression()
    print("the number of selected features are:",x_step.shape[1])
    reg.fit(trainX,trainY)
    reg.fit(x_step,trainY)
    y_pred_tr=reg.predict(x_step)
    y_pred_te=reg.predict(testX[step_data])
    e=evaluate(y_pred_tr,trainY,y_pred_te,testY)
    if i==0.5:
        rmse_tr.append(e[0])
        rmse_te.append(e[1])
        mae_tr.append(e[2])
        mae_te.append(e[3])

#build model for splines regression:
e=Spline_Regression(trainX,trainY,testX,testY)
rmse_tr.append(e[0])
rmse_te.append(e[1])
mae_tr.append(e[2])
mae_te.append(e[3])

#polt:
barWidth = 0.25
bars1 = rmse_tr
bars2 = rmse_te
bars3 = mae_tr
bars4 = mae_te
 # Set position 
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='rmse train')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='rmse test')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='mae train')
plt.bar(r4, bars4, color='#ff0000', width=barWidth, edgecolor='white', label='mae test')
plt.xlabel('method', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Polynominal', 'Ridge', 'Lasso', 'Elastic Net','backward','forward','splines'])
plt.legend()
plt.show()
plt.tight_layout()
plt.show()















    
