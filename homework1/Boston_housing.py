import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model,metrics
from sklearn.model_selection import train_test_split

#数据导入
data=pd.read_csv('.\housing.csv')
X=pd.read_csv('.\housing.csv',usecols=[*range(0,12)])
y=pd.read_csv('.\housing.csv',usecols=[*range(12,13)])

#线性回归
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg=linear_model.LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)

#模型验证，观察均方差和均方根差
y_pred=linreg.predict(X_test)
print("R-squred value:",metrics.r2_score(y_pred,y_test))
print("MAE:",metrics.mean_absolute_error(y_pred,y_test))
print("MSE:",metrics.mean_squared_error(y_pred,y_test))
print("RMSE",np.sqrt(metrics.mean_squared_error(y_pred,y_test)))

'''
#十折交叉验证
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=10)
print("R-squred value:",metrics.r2_score(y, predicted))
print("MAE:",metrics.mean_absolute_error(y, predicted))
print( "MSE:",metrics.mean_squared_error(y, predicted))
print( "RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))
'''
#作图
fig,ax=plt.subplots()
ax.scatter(y_test,y_pred)
ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
